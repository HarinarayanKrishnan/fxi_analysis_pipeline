import zmq
from mpi4py import MPI
import logging
import dxchange
from pathlib import Path
import h5py
import re
import os
import numpy as np
import tomopy
import argparse
from mpiarray import MpiArray, Distribution

from icxray.align_layers import find_angle
from icxray.align_tomo import apply_shifts, recon_shift_correction, recon_shift_correction_ltt,\
    recon_shift_correction
from icxray import utils, utils_mpi, align_tomo, align_layers

from ltt import ltt_tomopy

#FIXME: store options in an object that can be serialized!

# MPI specific code
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
# logging
logger = logging.getLogger(__name__)
utils_mpi.configure_logging(logger)
# abort all processes together on error
utils_mpi.configure_abort()


ncore = 10 # for Tomopy
DEBUG = True
peak_template = None
crop_center = False # speed things up by cropping center
crop_angles = False # remove certain angles, defaults to range [-70,70]
# general Spartan3 peak template
#peak_template = np.array([8, 8, 7, 8, 7, 6, 8, 7, 6, 8, 11, 11, 11, 11, 25, 19], np.int)
# peak template for y04_x13
#peak_template = np.array([11, 11, 7, 6, 9, 8, 6, 7, 6, 7, 10, 11, 12, 12, 24, 19], np.int)


import dxchange.reader as dxreader
ctx = zmq.Context()
socket = ctx.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")

def zmq_subscribe(uri, data_path, center_of_rotation):
    global socket
    global mpi_rank

    logger.info("HERE!")
    # socket = ctx.socket(zmq.SUB)
    # socket.setsockopt(zmq.SUBSCRIBE, b"")
    logger.info("connecting to ", uri, mpi_rank)

    socket.connect(uri)

    parameters = None
    tomo = {}
    darks = []
    bkg = []
    angles = []

    while True:
        msg = socket.recv_pyobj()
        # print(mpi_rank, msg)

        if "parameters" in msg:
            parameters = msg["parameters"]

        #if "tomo" in msg:
        #    tomo[msg["tomo_index"]] = msg["tomo"]

        if "dark_avg" in msg:
            dark_avg = msg["dark_avg"]

        if "white_avg" in msg:
            white_avg = msg["white_avg"]

        if "theta" in msg:
            theta = msg["thetas"]

    print(parameters, dark_avg.shape, white_avg.shape, tomo.shape)
    print("DONE!")


def read_nsls2_fxi18_h5(fname, proj=None, sino=None):
    """
    Read LNLS IMX standard data format.
    Parameters
    ----------
    fname : str
        Path to h5 file.
    proj : {sequence, int}, optional
        Specify projections to read. (start, end, step)
    sino : {sequence, int}, optional
        Specify sinograms to read. (start, end, step)
    Returns
    -------
    ndarray
        3D tomographic data.
    ndarray
        3D flat field data.
    ndarray
        3D dark field data.
    ndarray
        1D theta in radian.
    """
    tomo = dxreader.read_hdf5(fname, 'img_tomo', slc=(proj, sino), dtype=np.uint16)
    flats = dxreader.read_hdf5(fname, 'img_bkg', slc=(None, sino), dtype=np.uint16)
    darks = dxreader.read_hdf5(fname, 'img_dark', slc=(None, sino), dtype=np.uint16)
    theta = dxreader.read_hdf5(fname, 'angle', slc=(proj,))
    theta = np.deg2rad(theta)
    return tomo, flats, darks, theta


def read_nsls2_fxi18_h5_mpi(scan_id, data_path=".", size=None):
    global mpi_size
    filename = str(data_path/f"fly_scan_id_{scan_id}.h5")
    # get data size first, then load only data required, split on projs axis, 0
    with h5py.File(filename, "r") as f:
        projs_shape = f["/img_tomo"].shape
    distribution = Distribution.default(projs_shape, mpi_size)
    proj_slc = (distribution.offsets[mpi_rank], distribution.offsets[mpi_rank]+distribution.sizes[mpi_rank], 1)
    logger.info(f"{projs_shape}, {mpi_size}, {proj_slc}")

    #projs, flats, darks, thetas = dxchange.read_nsls2_fxi18_h5(filename, proj_slc)
    projs, flats, darks, thetas = read_nsls2_fxi18_h5(filename, proj_slc)
    # create mpi_projs and mpi_thetas.  The flats and darks are needed by all nodes.
    # NOTE: flats and darks should be uint16, but sometimes are float32
    flats = np.require(flats, dtype=np.uint16)
    darks = np.require(darks, dtype=np.uint16)
    if size is not None:
        slc1 = utils.center_slice(projs, size, axis=1)
        slc2 = utils.center_slice(projs, size, axis=2)
        projs = np.require(projs[:, slc1, slc2], requirements="C")
        flats = np.require(flats[:, slc1, slc2], requirements="C")
        darks = np.require(darks[:, slc1, slc2], requirements="C")
    mpi_projs = MpiArray(projs, distribution=distribution)
    mpi_thetas = MpiArray(thetas, distribution=distribution)
    return mpi_projs, flats, darks, mpi_thetas


def global_median_of_images_mpi(mpi_images):
    slices = mpi_images.scatter(1)
    image = np.median(slices, axis=0)
    return MpiArray(image, axis=0).allgather()


def normalize_mpi(mpi_projs, flats, darks):
    projs = mpi_projs.scatter(0)
    # use median instead of Tomopy mean to remove outliers
    flats = np.median(flats, axis=0, keepdims=True)
    darks = np.median(darks, axis=0, keepdims=True)
    projs = tomopy.normalize(projs, flats, darks)
    return MpiArray(projs)


def remove_low_frequency_mpi(mpi_data, w1=30, w2=100):
    imgs = mpi_data.scatter(0)
    for i in range(imgs.shape[0]):
        img = imgs[i]
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        dc = fshift[crow - w1:crow + w1, ccol - w1:ccol + w1]
        fshift[crow - w2:crow + w2, ccol - w2:ccol + w2] = 0
        fshift[crow - w1:crow + w1, ccol - w1:ccol + w1] = dc
        f_ishift = np.fft.ifftshift(fshift)
        img[:] = np.fft.ifft2(f_ishift)
        img[:] = np.abs(img)


def set_target_transmission_mpi(mpi_projs, mpi_thetas, target_transmission=0.90):
    # set a target transmission for zero degrees
    # remove constant offset of absorption (multiple each proj by correction factor)
    thetas = mpi_thetas.scatter(0)
    projs = mpi_projs.scatter(0)
    target_transmissions = utils.transmission(thetas, peak=target_transmission)
    for p in range(projs.shape[0]):
        projs[p] *= target_transmissions[p] / np.mean(projs[p])

#TODO: move to utils
def pad_sinos(sinos, pad=None):
    # pad sinos in x-axis for better recon with flat samples that extend past window
    if pad is None:
        pad = sinos.shape[2] // 2
    return np.pad(sinos, ((0, 0), (0, 0), (pad, pad)), "mean")


def remove_pad_rec(rec, pad=None):
    # remove padding from recon (on both y-axis and x-axis)
    # by default take center half
    if pad is None:
        pad = rec.shape[2] // 4
    xsize = rec.shape[2] - 2*pad
    slc_y = utils.center_slice(rec, xsize, axis=1)
    slc_x = utils.center_slice(rec, xsize, axis=2)
    return rec[:,slc_y,slc_x]


def tomopy_recon_mpi(mpi_projs, mpi_thetas, center_offset, algorithm="gridrec", start_z=None, end_z=None, pad=None, **kwargs):
    # perform MPI recon with padding on the X-axis to support flat samples
    # return mpi_rec
    # generate sinograms with padding
    mpi_sinos = utils_mpi.create_sinos_mpi(mpi_projs)
    sinos = mpi_sinos.scatter(0)
    sinos = pad_sinos(sinos, pad)
    gthetas = mpi_thetas.allgather() #global thetas
    center = sinos.shape[2] // 2 + center_offset
    rec = tomopy.recon(sinos, gthetas, center, True, algorithm, **kwargs)
    rec = remove_pad_rec(rec, pad)
    if start_z is not None and end_z is not None:
        rec = rec[:, start_z:end_z, :]
    mpi_rec = MpiArray(rec, distribution=mpi_sinos.distribution)
    return mpi_rec


def bnl_recon_fly_scan_h5_mpi(data_path, scan_id, center_offset=None):
    # load data from scan_id, from H5 or databroker
    # load data into MPI arrays, projs, flats, darks, metadata
    out_path = data_path / str(scan_id)
    if mpi_rank == 0:
        out_path.mkdir(exist_ok=True)
    comm.Barrier()
    os.chdir(str(out_path))
    logger.info(f"Load projections for {scan_id}")
    #FIXME: only loading center 1024 pixels!
    mpi_projs, flats, darks, mpi_thetas = read_nsls2_fxi18_h5_mpi(scan_id, data_path)#, 1024)
    #utils_mpi.write_stack_mpi(out_path/"raw", mpi_projs)

    do_work(out_path, scan_id, center_offset, mpi_projs, flats, darks, mpi_thetas)

def do_work(out_path, scan_id, center_offset, mpi_projs, flats, darks, mpi_thetas):

    logger.info("Flat Field Correction")
    mpi_projs = normalize_mpi(mpi_projs, flats, darks)
    #utils_mpi.write_stack_mpi(out_path/"flat", mpi_projs)
    del flats
    del darks

    logger.info("Outlier Removal")
    # TODO: base parameters on clean simulation data! - might need fill
    projs = mpi_projs.scatter(0)
    # TODO: put back in, I think it is causing issues right now...
    tomopy.remove_outlier(projs, 0.1, 5, ncore=ncore, out=projs)
    #tomopy.remove_outlier_cuda(projs, 0.1, 5, ncore, out=projs)
    np.clip(projs, 1E-6, 1-1E-6, projs)

    # TODO: distortion correction factor?

    # TODO: ring removal?

    # # flat field change correction
    # remove_low_frequency_mpi(mpi_projs)
    # utils_mpi.write_stack_mpi(out_path/"low_frequency_removed", mpi_projs)

    # bulk Si intensity correction
    # removes constant absorption contribution from bulk Si, and mounting material
    # TODO: base parameters on clean simulation data! - will need fill
    # TODO: alternatively, refine result after good recon - with theta offset
    target_transmission = 0.80
    logger.info(f"Setting target transmission to {target_transmission}")
    set_target_transmission_mpi(mpi_projs, mpi_thetas, target_transmission)
    projs = mpi_projs.scatter(0)
    np.clip(projs, 1E-6, 1-1E-6, projs)
    utils_mpi.write_stack_mpi(out_path/"constant_transmission", mpi_projs)

    # center finding - manual for now?
    if center_offset is None:
        logger.info("Finding center")
        # algorithm = "SART"
        # pixel_size = 2 * 0.000016 #16nm bin 1
        # options = {"PixelWidth": pixel_size,
        #            "PixelHeight": pixel_size,
        #            "windowFOV": False,
        #            "archDir": out_path,
        #            "_mpi_rank": mpi_rank,
        #            }
        # alg_params = {"N_iter": 1,
        #               "N_subsets": 20,
        #               "nonnegativityConstraint": True,
        #               "useFBPasSeedImage": False,
        #               # "Preconditioner": "RAMP",
        #               # "beta": 2e-7,
        #               # "p": 1,
        #               # "delta": 1/20, # delta sets edge strength (difference between regions divide by ten)
        #               # "inverseVarianceExponent": 1.0, # set to 1 to include noise model
        #               # "other": 3, #convergence of low frequencies
        #               }
        # # load data into LTT, then find the center before recon
        # ltt_tomopy.initialize_recon(sinos, thetas, xcenter, True, algorithm, options, ncore=ncore)
        # center = align_tomo.find_center_ltt(lambda c: ltt_tomopy.preview(center=c, algorithm=algorithm, sinogram_order=True, close=False, options=options, alg_params=alg_params, ncore=ncore), xcenter, 0.1, ratio=0.8)
        # ltt_tomopy.recon_close()
        logger.info("Padding sinos for center finding")
        mpi_sinos = utils_mpi.create_sinos_mpi(mpi_projs, ncore)
        sinos = mpi_sinos.scatter(0)
        sinos = pad_sinos(sinos)
        mpi_sinos = MpiArray(sinos)
        gthetas = mpi_thetas.allgather()
        xcenter = sinos.shape[2] // 2
        cen_range = (xcenter - 20, xcenter + 20, 0.5)
        if mpi_rank == mpi_size//2:
            tomopy.write_center(sinos, gthetas, out_path/("center"), cen_range, sinogram_order=True)
        del mpi_sinos, sinos
        import sys
        comm.Barrier()
        sys.exit()
        # center_offset = mpi_projs.shape[2]//2-center
        # mpi_projs.comm.Barrier() #for printing
        # print(f"{mpi_projs.mpi_rank}: center={center} offset={center_offset}")
    #center = xcenter + center_offset

    # Shift correction?
    # use MPI and binning for speed
    # use LTT for all recon
    # define recon extent with extra X and less Z

    # Quick recon - LTT with SART or other? (can't use FBP)
    algorithm = "gridrec"
    options = {
        "filter_name": "parzen",
    }
    logger.info(f"Finding layer alignment within volume using {algorithm}")
    mpi_rec = tomopy_recon_mpi(mpi_projs, mpi_thetas, center_offset, algorithm, ncore=ncore, **options)
    utils_mpi.write_stack_mpi(out_path/("quick_"+algorithm), mpi_rec)
    theta_deg, start_z1, end_z1 = align_layers.find_angle_mpi(mpi_rec, 0)
    logger.info(f"Theta offset angle {theta_deg:0.2} deg")
    # find phi angle (axis 2)
    phi_deg, start_z2, end_z2 = align_layers.find_angle_mpi(mpi_rec, 2)
    logger.info(f"Phi offset angle {phi_deg:0.2} deg")
    start_z = min(start_z1, start_z2)
    end_z = max(end_z1, end_z2)
    # add buffer for start and end
    start_z  = max(start_z-20, 0)
    end_z = min(end_z+20, mpi_rec.shape[1]-1)
    #FIXME: override start and end
    start_z = 0
    end_z = mpi_rec.shape[1]-1
    logger.info(f"Layer extent: {start_z} - {end_z}")

    # change theta with correction for next reconstruction
    thetas = mpi_thetas.scatter(0)
    thetas += np.deg2rad(theta_deg)
    # modify projections to remove phi angle
    # TODO: combine with stage shift code
    projs = mpi_projs.scatter(0)
    align_layers.apply_phi_correction(projs, thetas, phi_deg, projs)

    # Quick aligned recon
    algorithm = "gridrec"
    options = {
        "filter_name": "parzen",
    }
    logger.info("Quick Tomopy Recon")    
    mpi_rec = tomopy_recon_mpi(mpi_projs, mpi_thetas, center_offset, algorithm, ncore=ncore, **options)
    rec = mpi_rec.scatter(0)
    rec = rec[:,start_z:end_z,:]
    mpi_rec = MpiArray(rec)
    utils_mpi.write_stack_mpi(out_path/algorithm, mpi_rec)
    del mpi_rec, rec

    # Aligned recon
    # iterative recon with extra recon space in X and a restricted Z axis
    algorithm = "SART"#"RDLS"#"DFM"#"ASD-POCS"#"SART"#"FBP"
    logger.info(f"Reconstructing aligned layers using {algorithm}")
    mpi_sinos = utils_mpi.create_sinos_mpi(mpi_projs, ncore)
    #utils_mpi.write_stack_mpi(out_path/"sinos", mpi_sinos)
    sinos = mpi_sinos.scatter(0)
    # add padding to recon - fixes cupping effect
    xrecpadding = sinos.shape[2] // 2
    pixel_size = 2 * 0.000016  # 16nm bin 1
    options = {"PixelWidth": pixel_size,
               "PixelHeight": pixel_size,
               "ryoffset": start_z,
               "ryelements": end_z - start_z,
               "windowFOV": False,
               "rxelements": sinos.shape[2] + 2*xrecpadding,
               "rxoffset": -xrecpadding,
               "_mpi_rank": mpi_rank,
               }
    alg_params = {"N_iter": 50,
                   "nonnegativityConstraint": False,
                   "useFBPasSeedImage": False,
                   #"Preconditioner": "RAMP",
                   #"descentType": "CG",#"GD",
                   #"beta": 2e-7,
                   #"p": 1,
                   #"delta": 20/20, # delta sets edge strength (difference between regions divide by ten)
                   #"inverseVarianceExponent": 1.0, # set to 1 to include noise model
                   #"other": 3, #convergence of low frequencies
                  }
    #TODO: add support to add overlap in future with updates between iterations (see xray_trust6.py)
    gthetas = mpi_thetas.allgather() #global thetas
    center = sinos.shape[2] // 2 + center_offset
    if gthetas[1] < gthetas[0]:
        # decreasing angle, LTT doesn't support, switch data around
        # TODO: load in reversed order?
        gthetas = gthetas[::-1]
        sinos[:] = sinos[:,::-1,:]
    rec = ltt_tomopy.recon(sinos, gthetas, center, True, algorithm, alg_params, options, ncore=ncore)
    rec = rec[:, :, xrecpadding:xrecpadding+sinos.shape[2]]
    mpi_rec = MpiArray(rec, distribution=mpi_sinos.distribution)
    utils_mpi.write_stack_mpi(out_path/algorithm, mpi_rec)

    # Neural network processing?


    # Extract layers
    # Use template if available
    logger.info("Extracting Layers")
    mpi_layers = align_layers.extract_layers_mpi(mpi_rec)
    align_layers.write_layers_to_file_mpi(mpi_layers, "layers")


    logger.info(f"Finished {scan_id}")


if __name__ == "__main__":
    #DATA_PATH = Path("/run/media/sutherland/Elements/FXI_201902/FXI_2019_TA1_mosaic")
    DATA_PATH = Path("/NSLS2/xf18id1/users/2019Q3/SUTHERLAND_Proposal_304057")
    default_center_offset = None#-8.5  # used for center finding

    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers()

    sp_parser = sp.add_parser("h5", help="Read from h5 file path")
    sp_parser.add_argument("-d", "--directory", help="data directory", type=str, default=str(DATA_PATH))
    # sp_parser.add_argument("-c", "--center_offset", help="center of rotation offset", type=float, default=default_center_offset)
    sp_parser.add_argument("-id", "--scan_id", help="scan_id", type=int)
    sp_parser.add_argument("-c", "--center_offset", help="center of rotation offset", type=float, default=default_center_offset)

    sp_stream = sp.add_parser("stream", help="Read from ZMQ stream")
    sp_stream.add_argument("-s", "--stream_uri", help="stream_uri", required=True)
    sp_stream.add_argument("-c", "--center_offset", help="center of rotation offset", type=float, default=default_center_offset)
    sp_stream.add_argument("-d", "--directory", help="data directory", type=str, default=str(DATA_PATH))

    # read arguments from the command line
    args = parser.parse_args()
    var_args = vars(args)

    logger.info(f"Started with {mpi_size} processes")
    data_path = Path(args.directory)

    if "stream_uri" in var_args:
        logger.info("reading data from stream")
        zmq_subscribe(var_args["stream_uri"], var_args["directory"], var_args["center_offset"])
    else:
        if args.scan_id:
            bnl_recon_fly_scan_h5_mpi(data_path, args.scan_id, args.center_offset)
        else:
            filenames = sorted(data_path.glob("fly_scan_id_*.h5"))
            for filename in filenames:
                # extract ID number
                scan_id = int(re.findall('fly_scan_id_([0-9]+).h5', str(filename))[0])
                if scan_id < 11511:
                    continue
                logger.info(f"{scan_id}: Starting {filename}")
                bnl_recon_fly_scan_h5_mpi(data_path, scan_id, args.center_offset)

    logger.info("Done!")
