import logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR) #suppress import logging


import sys
#NOTE: hack to set LTT path
sys.path.insert(0,"/home/sutherland/ltt/v1.6.2")
sys.path.insert(0,"/home/sutherland/ltt/v1.6.2/python")
import ltt_recon

import os
import numpy as np
from matplotlib import pyplot as plt
import psutil
import concurrent.futures as cf
from scipy.ndimage.interpolation import zoom, shift, rotate
from skimage.feature import register_translation, canny, peak_local_max
from skimage.exposure import equalize_adapthist, rescale_intensity
from scipy.optimize import minimize
#from sklearn.preprocessing import scale
#import cv2

import dxchange
import tomopy
from utils import write_stack, load_stack, center_slices, zoom_array
from align_layers import find_angle
from align_tomo import apply_shifts, recon_shift_correction, recon_shift_correction_ltt,\
    recon_shift_correction, proj_shift_correction

# configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
                    level=logging.INFO)
root_logger.setLevel(logging.INFO) # could also set to WARNING...
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# default settings...
crop_angles = False
theta_offset_deg = 0
crop_center = None
center_offset = None
align_flat_sample = True
zoom_level = None
pixel_size = 0.000020

# working directory
path = "/home/sutherland/xray/dmea/TA1_baseline"
filename = os.path.join(path, "TA1_LFOV_b1_90000ms_y0_x-2.txrm")
bg_filename = os.path.join(path, "TA1_LFOV_b1_90000ms_y 0_x-2_5_BG.txrm")
center_offset = 2#None#-9
zoom_level = 0.5

# create scan folder
if not os.path.exists(path):
    os.mkdir(path)

os.chdir(path)


logger.info("Load projections")
projs, metadata = dxchange.reader.read_txm(filename)
flats = dxchange.reader.read_txm(bg_filename)[0]
darks = np.zeros((1, flats.shape[1], flats.shape[2]), flats.dtype)
thetas = metadata["thetas"]
xpos = metadata["x_positions"][0]
ypos = metadata["y_positions"][0]
shifts = np.array((metadata['y-shifts'], metadata['x-shifts'])).T
shifts[:, 0] *= -1  # y-shifts are inverted!
shifts = np.require(shifts, requirements="C")
logger.info(f"Tomo taken at x:{xpos:0.2f}; y:{ypos:0.2f}")
thetas_deg = np.rad2deg(thetas)
logger.info(f"Theta range is {thetas_deg[0]:0.2f} - {thetas_deg[-1]:0.2f}")

if crop_angles:
    # remove angles not between -70 and 70
    start_index = 0
    end_index = 0
    for i, theta in enumerate(thetas_deg):
        if theta < -70:
            start_index = i
        if theta < 70:
            end_index = i  
    slc = np.s_[start_index:end_index]
    thetas = thetas[slc]
    thetas_deg = thetas_deg[slc]
    projs = projs[slc]
#   
# #write_stack("raw", projs)
#         
# read flats and darks, do flat field correction
logger.info("flat field correction..")
flat = np.median(flats, axis=0)
dark = np.median(darks, axis=0)
del flats, darks
       
# Only use middle of image
if crop_center is not None:
    logger.info("Cropping to center of image")
    slc = center_slices(projs[0], [crop_center, crop_center])
    projs = projs[:, slc[0], slc[1]]
    dark = dark[slc]
    flat = flat[slc]
  
projs = tomopy.normalize(projs, flat[np.newaxis,:,:], dark[np.newaxis,:,:]) 
logger.info("Remove outliers")
tomopy.remove_outlier(projs, 0.1, out=projs)

# logger.info("shift correction using projections")
# shifts = proj_shift_correction(projs)
#print(shifts)
#np.savetxt("shifts_proj.txt", shifts)
#apply_shifts(projs, shifts, out=projs)
#shifted_projs = projs #change name after shifts
#del projs
#write_stack("shifted_projs_using_elastix", shifted_projs)

# initial shifts
#shifts = np.zeros((projs.shape[0], 2), np.float32)
#shifts = np.loadtxt("shifts_ltt.txt")
 
# # align using image correlation
# def process_img(img, crop=(20, 400)):
#     clahe = cv2.createCLAHE(clipLimit=0.0, tileGridSize=(12,12))
#     img = rescale_intensity(img, out_range=np.uint16)
#     img = img.astype(np.uint16)
#     img = clahe.apply(img)
#     img = img.astype(np.float32)
#     img /= 2^16-1
#     #img = equalize_adapthist(img, 12)
#     img -= np.mean(img)
#     img /= np.std(img)
#     img = img[crop[0]:-crop[0], crop[1]:-crop[1]]
#     return img
#  
# for i in range(projs.shape[0] - 1):
#     translation = register_translation(process_img(projs[i]), process_img(projs[i+1]), 1)[0]
#     logger.info("%03d: %s"%(i, str(translation)))
#     shifts[i+1] = translation
# np.cumsum(shifts, axis=0, out=shifts)
# shifts[:, 0] -= np.mean(shifts[:,0])
# shifts[:, 1] -= np.mean(shifts[:,1])
# np.savetxt("shifts_correlation.txt", shifts)

#shifts = np.loadtxt("shifts_correlation.txt")


logger.info("Align projections using recon")
alg = "SART"
options = {"PixelWidth": pixel_size,
           "PixelHeight": pixel_size,
           "windowFOV": True,
           }
alg_params = {"N_iter": 50,
              "N_subsets": 20,
              "nonnegativityConstraint": False,
              "useFBPasSeedImage": False,
              }

xcenter = projs.shape[2] // 2
center = xcenter
if center_offset is not None:
    center = xcenter + center_offset
#scales=(8,)*16 + (8, 4, 2)
#shift_iters=(1,)*16 + (8, 8, 32)
scales=(8,)*16 + (4,)*8 + (2,)*8
shift_iters=(1,)*16 + (1,)*8 + (1,)*8
shifts, rec, center = recon_shift_correction_ltt(projs, thetas, center, init_shifts=shifts, scales=scales, shift_iters=shift_iters, alg=alg, alg_params=alg_params, options=options)
logger.info(f"center={center}")
np.savetxt("shifts_ltt.txt", shifts)
write_stack("shifted_rec", rec)
del rec

if zoom_level:
    logger.info("Zoom images by {zoom_level}x")
    projs = zoom_array(projs, zoom_level)
    # NOTE: must scale center offset and pixel size
    center_offset *= zoom_level
    pixel_size *= zoom_level
    shifts *= zoom_level

# # recon
apply_shifts(projs, shifts, out=projs)
shifted_projs = projs #change name after shifts
del projs
write_stack("shifted_projs", shifted_projs)
sinos = tomopy.init_tomo(shifted_projs, sinogram_order=False, sharedmem=False)
tomopy.minus_log(sinos, out=sinos)

# logger.info("Finding center of rotation")
xcenter = sinos.shape[2] // 2
center = xcenter
if center_offset is not None:
    center = xcenter + center_offset
else:
    tomopy.write_center(sinos, thetas, "xcenter", (xcenter - 50, xcenter + 50, 1), sinogram_order=True)
    #center = tomopy.find_center(sinos, thetas, tol=0.1, mask=True, ratio=0.8, sinogram_order=True, algorithm="sirt", num_iter=10)
    #tomopy.write_center(sinos, thetas, "center", (center - 20, center + 20, 0.5), sinogram_order=True, algorithm="sirt", num_iter=10)
logger.info(f"center: {center:0.2f}")


# for flat samples, align recon to surface
if align_flat_sample:
    # Theta angle correction, do another recon to get better data afterwards!
    alg = "FBP"
    logger.info(f"Correcting theta using recon with {alg}")
    options = {"PixelWidth": pixel_size,
               "PixelHeight": pixel_size,
               "loggingLevel": "logDEBUG",
               }
    alg_params = {"extrapolate": True,
                  }
    rec = ltt_recon.recon(sinos,
                          thetas,
                          center,
                          True, #sinogram_order
                          alg,
                          alg_params,
                          options)
    del sinos
    # find theta angle (axis 0)
    theta_deg = find_angle(rec, 0)
    logger.info(f"Theta angle {theta_deg:0.2f} deg")
    # find phi angle (axis 2)
    phi_deg = find_angle(rec, 2)
    logger.info(f"Phi angle {phi_deg:0.2f} deg")
    del rec
    
    # change theta with correction for next reconstruction
    thetas_deg += theta_deg
    thetas += theta_deg * np.pi / 180.

    # modify projections to remove phi angle
    # TODO: combine with shifts?
    for p in range(shifted_projs.shape[0]):
        theta = thetas[p]
        proj = shifted_projs[p]
        for r in range(proj.shape[0]):
            r_offset = r - proj.shape[0] // 2
            x_shift = -r_offset * np.tan(phi_deg * np.pi / 180) * np.sin(theta)
            shift(proj[r], x_shift, proj[r], mode='nearest')

    write_stack("phi_corrected", shifted_projs)
    # re-create sinos
    sinos = tomopy.init_tomo(shifted_projs, sinogram_order=False, sharedmem=False)
    tomopy.minus_log(sinos, out=sinos)

del shifted_projs
 
alg = "gridrec"
options = {}
logger.info(f"Reconstruct using {alg}")
rec = tomopy.recon(sinos, thetas, center, sinogram_order=True, algorithm=alg, **options)
write_stack(alg, rec)
del rec

alg = "SART"
#rec = load_stack(alg)
rec = None
iterations = 0
iter_per_loop = 10
while(True):
    logger.info(f"iterations: {iterations}")
    iterations += iter_per_loop
    options = {"PixelWidth": pixel_size,
               "PixelHeight": pixel_size,
               "windowFOV": False, #circular crop disabled
               }
    alg_params = {"N_iter": iter_per_loop,
                  "N_subsets": 20,
                  "nonnegativityConstraint": False,
                  "useFBPasSeedImage": False,
                  #"Preconditioner": "RAMP",
                  #"beta": 2e-7,
                  #"p": 1,
                  #"delta": 1/20, # delta sets edge strength (difference between regions divide by ten)
                  #"inverseVarianceExponent": 1.0, # set to 1 to include noise model
                  #"other": 3, #convergence of low frequencies
                  }
    logger.info(f"Reconstruct using {alg}")
    rec = ltt_recon.recon(sinos,
                          thetas,
                          center,
                          True, #sinogram_order
                          alg,
                          alg_params,
                          options,
                          rec)
    write_stack(alg, rec)
logger.info("done")

