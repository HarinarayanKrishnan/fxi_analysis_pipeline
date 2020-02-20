import logging
import os
import shutil
import numpy as np
import dxchange
import tomopy
import psutil
#from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import zoom, shift, rotate
#from scipy.ndimage.filters import laplace, gaussian_filter
from skimage.feature import register_translation, canny, peak_local_max
#from skimage.exposure import equalize_adapthist
#from skimage.filters.rank import gradient
#from skimage.morphology import disk
from scipy.optimize import minimize
import concurrent.futures as cf
#import SimpleITK as sitk
#from .utils import center_slice

#sitk.ProcessObject.SetGlobalDefaultDebug(False)

from ltt import ltt_tomopy


logger = logging.getLogger(__name__)

def write_stack(path, projs):
    logger.info("writing to %s"%path)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    dxchange.writer.write_tiff_stack(projs, "%s/img"%(path), overwrite="True")
    

def apply_shifts(projs, shifts, ncore=None, out=None):
    if not ncore:
        ncore = psutil.cpu_count(True)
    if out is None:
        out = np.empty(projs.shape, dtype=projs.dtype)
    # TODO: figure out a better way to do this...
    if ncore == 1:
        # just do it here directly
        for i in range(projs.shape[0]):
            shift(projs[i], shifts[i], out[i], mode='nearest')
    else:
        with cf.ThreadPoolExecutor(ncore) as e:
            futures = [e.submit(shift, projs[i], shifts[i], out[i], mode='nearest') for i in range(projs.shape[0])]
            cf.wait(futures)
    return out


def zoom_array(proj, zoom_level=0.5, ncore=None, **kwargs):
    if not ncore:
        ncore = psutil.cpu_count(True)
    output = np.empty((proj.shape[0], int(round(proj.shape[1]*zoom_level)), int(round(proj.shape[2]*zoom_level))), dtype=proj.dtype)
    mulargs = zip(proj, [zoom_level] * proj.shape[0], output)
    with cf.ThreadPoolExecutor(ncore) as e:
        futures = [e.submit(zoom, *args, **kwargs) for args in mulargs]
        cf.wait(futures)
    return output


#FIXME: center shift also shifts in X!
def center_shifts(shifts, thetas, out=None):
    # center shifts in Y and Z recon
    if out is None:
        out = np.copy(shifts)
    # for Y shifts, average them to be zero.  This prevents drifting up and down
    out[:,0] -= np.mean(out[:,0])
    # for Z recon shifts, remove Z recon component.
    # this prevents the reconstruction from shifting in Z (forwards and backwards)
    # remove x-shift from z-shift calculation
    center_offset = np.mean(shifts[:,1])
      
    def _cost(args):
        x_shift, z_shift = args
        new_shifts = shifts[:,1] - z_shift * np.sin(thetas) - x_shift * np.cos(thetas)
        new_center_offset = np.mean(new_shifts)
        return np.sum(np.abs(new_shifts - new_center_offset + center_offset))
      
    options = {'ftol':0.0001}
    #options = None
    res = minimize(
        _cost, 
        (0.0, 0.0),
        method='Powell',
        #tol=0.01,
        options=options)
    x_shift, z_shift = res.x
    logger.info("x: %0.2f, z: %0.2f", x_shift, z_shift)
    out[:,1] -= z_shift * np.sin(thetas) + x_shift * np.cos(thetas)
    out[:,1] -= np.mean(out[:,1])
    return out


# def center_shifts(shifts, thetas, out=None):
#     # center shifts in Y and Z recon
#     if out is None:
#         out = np.copy(shifts)
#     # for Y shifts, average them to be zero.  This prevents drifting up and down
#     out[:,0] -= np.mean(out[:,0])
#     # for Z recon shifts, remove Z recon component.
#     # this prevents the reconstruction from shifting in Z (forwards and backwards)
#     # remove x-shift from z-shift calculation
#     x_shift = np.mean(shifts[:,1])
#      
#     def _cost(args):
#         z_shift = args
#         return np.sum(np.abs(shifts[:,1] - x_shift - z_shift * np.sin(thetas)))
#      
#     options = {'ftol':0.0001}
#     #options = None
#     res = minimize(
#         _cost, 
#         0.0,
#         method='Powell',
#         #tol=0.01,
#         options=options)
#     z_shift = res.x
#     logger.info("x: %0.2f, z: %0.2f", x_shift, z_shift)
#     out[:,1] -= z_shift * np.sin(thetas)
#     return out


def proj_shift_correction(projs):
    # register projections to get shifts
    shifts = np.zeros((projs.shape[0], 2), np.float32)
    prev = None
    for p, proj in enumerate(projs):
        # create copy
        proj = np.copy(proj)
        #proj -= gaussian_filter(proj, sigma=5)
        if prev is not None:
            shifts[p] = find_translation(proj, prev)
            #shifts[p] = register_translation(proj, prev, 100)[0]
        prev = proj
    np.cumsum(shifts, 0, out=shifts)
    shifts -= np.mean(shifts, axis=0)
    return shifts


#FIXME: alg => algorithm
def recon_shift_correction(projs, thetas, center=None, init_shifts=None, init_recon=None, scales=(8,), shift_iters=(16,), alg="gridrec", **kwargs):
    shifts = init_shifts
    if center is None:
        center = projs.shape[2]/2. #center of rotation in un-scaled pixels
    for scale, shift_iter in zip(scales, shift_iters):
        rec = init_recon
        for i in range(shift_iter): 
            logger.info("Shift correction, scale %d: %d of %d" % (scale, i+1, shift_iter))
            shifts, rec, center = shift_correction(projs, thetas, center, shifts, scale, alg, init_recon=rec, **kwargs)
            np.savetxt("recon_shifts_%d_%d.txt"%(scale, i), shifts)
    return shifts, rec, center


def shift_correction(projs, thetas, center=None, init_shifts=None, scale=1, alg="gridrec", init_recon=None, **kwargs):
    # initial shift values
    if init_shifts is None:
        shifts = np.zeros((projs.shape[0], 2))
    else:
        shifts = np.copy(init_shifts)
    # scaled projections used after recon/reproject
    if scale != 1:
        scaled_projs = zoom_array(projs, 1.0 / scale)
    else:
        scaled_projs = projs
    if center is None:
        center = projs.shape[2] / 2.
    center /= scale
    # perform initial shifts
    shifted_projs = apply_shifts(projs, shifts)
    # scale shifted if necessary
    if scale != 1:
        shifted_projs = zoom_array(shifted_projs, 1.0 / scale)
    np.clip(shifted_projs, 0, 1.0, shifted_projs)
    tomopy.minus_log(shifted_projs, out=shifted_projs)
    # find center of rotation
    logger.info("finding center...")
    center = tomopy.find_center(shifted_projs, thetas, tol=0.01, init=center, algorithm=alg, **kwargs)
    logger.info("Updated center to be %0.3f", center*scale)
    # recon
    logger.info("Shift reconstruct using %s"%alg)
    rec = init_recon
    rec = tomopy.recon(shifted_projs, thetas, center, sinogram_order=False, algorithm=alg, init_recon=rec, **kwargs)
    del shifted_projs
    np.clip(rec, 0.0, 1.0, rec) #TODO: needed?
    # simulate projections
    sim_projs = tomopy.project(rec, thetas, center, pad=False, emission=False)
    write_stack("test_rec", rec)
    write_stack("sim_projs", sim_projs)
    # calculate shift for each
    translation_sum = np.zeros((2,))
    logger.info("Projecting and aligning slices")
    for t in range(sim_projs.shape[0]):
        translation = register_translation(sim_projs[t], scaled_projs[t], 100)[0]
        translation_sum += np.abs(shifts[t] - translation*scale)
        shifts[t] = translation*scale
    logger.info("translation sum is y:%0.2f, x:%0.2f"%(translation_sum[0], translation_sum[1]))
    del scaled_projs
    del sim_projs
    return shifts, rec, center*scale


def recon_shift_correction_ltt(projs, thetas, center=None, init_shifts=None, init_recon=None, scales=(8,), shift_iters=(16,), alg="FBP", **kwargs):
    shifts = init_shifts
    if center is None:
        center = projs.shape[2]/2. #center of rotation in un-scaled pixels
    kwargs.setdefault("alg_params", dict())
    init_seed_image = kwargs["alg_params"].get("seedImage")
    for scale, shift_iter in zip(scales, shift_iters):
        rec = init_recon
        init_recon = None # can't use init_recon for other scales
        for i in range(shift_iter): 
            if alg != "FBP":
                if rec is not None:
                    kwargs["alg_params"]["seedImage"] = "mem"
                elif init_seed_image is not None:
                    kwargs["alg_params"]["seedImage"] = init_seed_image
                else:
                    kwargs["alg_params"].pop("seedImage", None)
            logger.info("Shift correction LTT, scale %d: %d of %d" % (scale, i+1, shift_iter))
            shifts, rec, center = shift_correction_ltt(projs, thetas, center, shifts, scale, alg, find_center=(i==0), close=False, out=rec, **kwargs)
            np.savetxt("recon_shifts_ltt_%d_%d.txt"%(scale, i), shifts)
        ltt_tomopy.recon_close() # delete data from LTT
    return shifts, rec, center


def shift_correction_ltt(projs, thetas, center, init_shifts=None, scale=1, alg="FBP", find_center=False, close=True, **kwargs):
    if init_shifts is None:
        shifts = np.zeros((projs.shape[0], 2))
    else:
        shifts = np.copy(init_shifts)
    # shift correction loop with scaling
    if scale != 1:
        scaled_projs = zoom_array(projs, 1.0 / scale)
    else:
        scaled_projs = projs
    if center is None:
        center = projs.shape[2] / 2.
    center /= scale
    # apply shifts first (would be more efficient to do second, but this is more accurate)
    shifted_projs = apply_shifts(projs, shifts)
    write_stack("shifted_projs_ltt", shifted_projs)
    # scale if necessary
    if scale != 1:
        shifted_projs = zoom_array(shifted_projs, 1.0 / scale)
    np.clip(shifted_projs, 0, 1.0, shifted_projs)
    write_stack("shifted_projs_scaled_ltt", shifted_projs)

    # center finding and recon
    options = kwargs.get("options")
    if options is not None:
        options = options.copy()
        if "PixelWidth" in options:
            options["PixelWidth"] /= scale
        if "PixelHeight" in options:
            options["PixelHeight"] /= scale
    # load data into LTT, then find the center before recon
    ltt_tomopy.initialize_recon(shifted_projs, thetas, center, False, alg, **kwargs)
    logger.info("finding center...")
    # use "try" function in LTT to find best center
    if find_center:
        center = find_center_ltt(lambda c: ltt_tomopy.preview(center=c, algorithm=alg, close=False, **kwargs), center, 0.1, ratio=0.8)
    logger.info("Updated center to be %0.3f", center*scale)
    logger.info("Reconstruct using %s"%alg)
    translation_sum = np.zeros((2,))
    ltt_tomopy.recon_close()
    # recon and project each angle separately...
    bad_data_filename = "badDataMap.tif"
    alg_params = kwargs.get("alg_params", {}).copy()
    alg_params["badDataFile"] = bad_data_filename
    kwargs["alg_params"] = alg_params
    for t in range(thetas.shape[0]):
        logger.info(f"theta index {t+1} of {thetas.shape[0]}")
        #remove projection and angle from thetas
        bad_data = np.zeros((shifted_projs.shape[0], shifted_projs.shape[2]), np.float32)
        bad_data[t,:] = 1
        dxchange.write_tiff(bad_data, bad_data_filename, overwrite=True)
        ltt_tomopy.LTT.writeImageFile(bad_data, bad_data_filename, 2.0, options["PixelWidth"])
        rec = ltt_tomopy.recon(shifted_projs, thetas, center, False, alg, close=False, **kwargs)
        # simulate projections
        write_stack("test_rec_ltt", rec)
        # TODO: only project needed view?
        alg_params = None#{'views':f'({t},{t})'}
        sim_projs = ltt_tomopy.project(emission=False, alg_params=alg_params, close=True)
        # calculate shift
        translation = find_translation(sim_projs[t], scaled_projs[t], 100)
        translation_sum += np.abs(shifts[t] - translation*scale)
        #shifts[t] = (shifts[t] + translation*scale) / 2.0
        shifts[t] = translation*scale
    logger.info("Translation sum is y:%0.2f, x:%0.2f"%(translation_sum[0], translation_sum[1]))
    del scaled_projs
    if close:
        ltt_tomopy.recon_close() # delete data from LTT
    return shifts, rec, center*scale


# def shift_correction_ltt(projs, thetas, center, init_shifts=None, scale=1, alg="FBP", find_center=False, close=True, **kwargs):
#     if init_shifts is None:
#         shifts = np.zeros((projs.shape[0], 2))
#     else:
#         shifts = np.copy(init_shifts)
#     # shift correction loop with scaling
#     if scale != 1:
#         scaled_projs = zoom_array(projs, 1.0 / scale)
#     else:
#         scaled_projs = projs
#     if center is None:
#         center = projs.shape[2] / 2.
#     center /= scale
#     # apply shifts first (would be more efficient to do second, but this is more accurate)
#     shifted_projs = apply_shifts(projs, shifts)
#     write_stack("shifted_projs_ltt", shifted_projs)
#     # scale if necessary
#     if scale != 1:
#         shifted_projs = zoom_array(shifted_projs, 1.0 / scale)
#     np.clip(shifted_projs, 0, 1.0, shifted_projs)
#     write_stack("shifted_projs_scaled_ltt", shifted_projs)
#
#     # center finding and recon
#     options = kwargs.get("options")
#     if options is not None:
#         options = options.copy()
#         if "PixelWidth" in options:
#             options["PixelWidth"] /= scale
#         if "PixelHeight" in options:
#             options["PixelHeight"] /= scale
#     # load data into LTT, then find the center before recon
#     ltt_recon.initialize_recon(shifted_projs, thetas, center, False, alg, **kwargs)
#     logger.info("finding center...")
#     # use "try" function in LTT to find best center
#     if find_center:
#         center = find_center_ltt(lambda c: ltt_recon.preview(center=c, algorithm=alg, close=False, **kwargs), center, 0.1, ratio=0.8)
#     logger.info("Updated center to be %0.3f", center*scale)
#     logger.info("Reconstruct using %s"%alg)
#     rec = ltt_recon.recon(center=center, algorithm=alg, close=False, **kwargs)
#     del shifted_projs
#     # simulate projections
#     sim_projs = ltt_recon.project(emission=False)
#     write_stack("test_rec_ltt", rec)
#     write_stack("sim_projs_ltt", sim_projs)
#     # calculate shift for each
#     translation_sum = np.zeros((2,))
#     logger.info("Projecting and aligning slices")
#     for t in range(sim_projs.shape[0]):
#         translation = find_translation(sim_projs[t], scaled_projs[t], 100)
#         translation_sum += np.abs(shifts[t] - translation*scale)
#         #shifts[t] = (shifts[t] + translation*scale) / 2.0
#         shifts[t] = translation*scale
#     logger.info("Translation sum is y:%0.2f, x:%0.2f"%(translation_sum[0], translation_sum[1]))
#     del scaled_projs
#     del sim_projs
#     if close:
#         ltt_recon.recon_close() # delete data from LTT
#     return shifts, rec, center*scale


def find_translation(canvas, template, resolution=100):
    # create image stack
    #images = np.stack((canvas, template), 0)
    # CSLabs registration technique
    # blur edges and center
    # NOTE: creates copy!
    #images = tomopy.blur_edges(images)
    #plt.imshow(images[0])
    #plt.show()
    #plt.imshow(images[1])
    #plt.show()
    # take derivative (laplacian)
    #laplace(images[0], images[0])
    #laplace(images[1], images[1])
    #plt.imshow(images[0])
    #plt.show()
    #plt.imshow(images[1])
    #plt.show()
    # do correlation
    #translation = register_translation(images[0], images[1], resolution)[0]

    # Elastix (recommended by LANL)
    # parameterMap = sitk.GetDefaultParameterMap('translation')
    # elastixImageFilter = sitk.ElastixImageFilter()
    # elastixImageFilter.LogToConsoleOff()
    # elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(images[0]))
    # elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(images[1]))
    # elastixImageFilter.SetParameterMap(parameterMap)
    # elastixImageFilter.Execute()
    # transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    # #NOTE: elastix returns (x,y) shifts and we want (y,x)
    # translation2 = np.array(transformParameterMap[0]["TransformParameters"][::-1], np.float32)

    translation = register_translation(canvas, template, resolution)[0]
    return translation


def find_center_ltt(recon_function, init, tol=0.5, mask=True, ratio=1.):
    """
    Find rotation axis location.

    The function exploits systematic artifacts in reconstructed images
    due to shifts in the rotation center. It uses image entropy
    as the error metric and ''Nelder-Mead'' routine (of the scipy
    optimization module) as the optimizer :cite:`Donath:06`.

    Parameters
    ----------
    recon_function : function
        function that takes center as argument and performs recon
    init : float
        Initial guess for the center.
    tol : scalar
        Desired sub-pixel accuracy.
    mask : bool, optional
        If ``True``, apply a circular mask to the reconstructed image to
        limit the analysis into a circular region.
    ratio : float, optional
        The ratio of the radius of the circular mask to the edge of the
        reconstructed image.
    Returns
    -------
    float
        Rotation axis location.
    """
    # extract slice we are using to find center
    hmin, hmax = _adjust_hist_limits(recon_function(init), mask, ratio)

    # Magic is ready to happen...
    cache = {} #minimizer will re-evaluate values, use cache
    res = minimize(
        _find_center_cost, init,
        args=(recon_function, hmin, hmax, mask, ratio, cache),
        method='Nelder-Mead',
        tol=tol)
    return res.x


def _adjust_hist_limits(rec, mask, ratio):
    # Apply circular mask.
    if mask is True:
        rec = tomopy.circ_mask(rec, axis=0, ratio=ratio)

    # Adjust histogram boundaries according to reconstruction.
    return _adjust_hist_min(rec.min()), _adjust_hist_max(rec.max())


def _adjust_hist_min(val):
    if val < 0:
        val = 2 * val
    elif val >= 0:
        val = 0.5 * val
    return val


def _adjust_hist_max(val):
    if val < 0:
        val = 0.5 * val
    elif val >= 0:
        val = 2 * val
    return val


def _find_center_cost(center, recon_function, hmin, hmax, mask, ratio, cache):
    """
    Cost function used for the ``find_center`` routine.
    """
    logger.info('Trying rotation center: %s', center)
    center = np.array(center, dtype='float32')
    if float(center) in cache:
        logger.info("Using cached value for center: %s", center)
        return cache[float(center)]
    rec = recon_function(center)

    if mask is True:
        rec = tomopy.circ_mask(rec, axis=0, ratio=ratio)

    hist, _ = np.histogram(rec, bins=64, range=[hmin, hmax])
    hist = hist.astype('float32') / rec.size + 1e-12
    val = -np.dot(hist, np.log2(hist))
    logger.info("Function value = %f", val)
    cache[float(center)] = val
    return val


def test_find_translation():
    # test if find_translation is working correctly
    # load sample tomopy image
    # shift image by some amount
    # verify translation is found
    shifts = (10.1111, -5.2222)
    canvas = tomopy.barbara(512)[0]
    template = shift(canvas, shifts)
    calc_shifts = find_translation(canvas, template, 100)
    print(f"shifts={shifts}, calculated shifts={calc_shifts}")

if __name__ == "__main__":
    test_find_translation()