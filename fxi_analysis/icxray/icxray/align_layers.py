from mpi4py import MPI
from mpiarray import MpiArray
import logging
import os
import numpy as np
from scipy.ndimage.interpolation import rotate, shift
from scipy.ndimage import filters
from scipy.optimize import minimize
from scipy.signal import argrelmax, argrelmin
import dxchange

logger = logging.getLogger(__name__)


def mad(a, c=.6745, axis=0, center=np.median):
    # c \approx .6745
    """
    The Median Absolute Deviation along given axis of an array

    Parameters
    ----------
    a : array-like
        Input array.
    c : float, optional
        The normalization constant.  Defined as scipy.stats.norm.ppf(3/4.),
        which is approximately .6745.
    axis : int, optional
        The default is 0. Can also be None.
    center : callable or float
        If a callable is provided, such as the default `np.median` then it
        is expected to be called center(a). The axis argument will be applied
        via np.apply_over_axes. Otherwise, provide a float.

    Returns
    -------
    mad : float
        `mad` = median(abs(`a` - center))/`c`
    """
    a = np.asarray(a)
    if callable(center):
        center = np.apply_over_axes(center, a, axis)
    return np.median((np.fabs(a-center))/c, axis=axis)


def find_angle_from_layer_diff(layer_diff, axis=0):
    # find angle of layers by maximizing variance
    
    def _angle_cost(angle_deg, layer_diff):
        angle_deg = angle_deg[0]
        rotated = rotate(layer_diff, angle_deg, reshape=False)
        cost = -np.var(np.sum(rotated, axis=axis))
        logger.debug("%0.3f: %0.3f"%(angle_deg, cost))
        return cost

    options = {'xtol':0.0001,
               }

    res = minimize(
        _angle_cost, 
        0.0,
        args=(layer_diff),
        method='Powell',
        #tol=0.1,
        options=options)
    angle_deg = res.x
    return angle_deg


def find_extent_from_layer_diff(layer_diff, axis=0):
    line = np.mean(layer_diff, axis)
    filters.gaussian_filter1d(line, 3, output=line)
    line = np.abs(line)
    std_val = np.std(line)
    # find beginning and ending index
    for start in range(line.shape[0]):
        if line[start] > 2*std_val:
            break
    for end in range(line.shape[0] - 1, 0, -1):
        if line[end] > 2*std_val:
            break
    logger.debug(f"Extent: {start-1} {end+1}")
    return max(start-1, 0), min(end+1, line.shape[0]-1)


def find_angle_mpi(mpi_rec, axis=0):
    from mpiarray import MpiArray
    mpi_rec_diff = mpi_rec.copy(deep=True)
    # take different between layers to measure angle
    rec_diff = mpi_rec_diff.scatter(0, 5) # add padding
    rec_diff[rec_diff<0] = 0
    # smooth data between IC layers and take difference
    for i in range(rec_diff.shape[1]):
        filters.gaussian_filter(rec_diff[:,i], (9, 9), output=rec_diff[:,i])
        if i > 0:
            rec_diff[:,i-1] -= rec_diff[:,i]
    rec_diff = mpi_rec_diff.scatter(0) # remove padding

    # now collapse axis to correct angle (already dist on zero)
    if axis == 0:
        layer_diff = np.sum(rec_diff, axis=0, keepdims=True)
        layer_diff = MpiArray(layer_diff).allgather()
        layer_diff = np.sum(layer_diff, axis=axis)
        layer_diff /= rec_diff.shape[0]
    else:
        layer_diff = np.sum(rec_diff, axis=axis)
        layer_diff = MpiArray(layer_diff).allgather()
        layer_diff = np.rot90(layer_diff, axes=(1,0)) #TODO: currently specific to axis=2

    # smooth out layer_diff
    filters.gaussian_filter(layer_diff, (3,3), output=layer_diff)
    # remove sides
    side_trim = layer_diff.shape[0]//4
    layer_diff = layer_diff[side_trim:-side_trim]

    if mpi_rec.mpi_rank == 0:
        # TODO: need to return layer_diff or ignore, don't write to disk
        dxchange.write_tiff(layer_diff, fname='layer_diff_%d.tiff'%axis, overwrite=True)
        angle_deg = find_angle_from_layer_diff(layer_diff, axis=1) #FIXME: axis
        # find where the layer starts and stops
        rotated_layer_diff = rotate(layer_diff, angle_deg, reshape=False)
        start, end = find_extent_from_layer_diff(rotated_layer_diff, axis=1)
        # NOTE: don't forget to add back on the trimmed off amount from earlier!
        start += side_trim
        end += side_trim
    else:
        angle_deg, start, end = None, None, None
    angle_deg = mpi_rec.comm.bcast(angle_deg, root=0)
    start = mpi_rec.comm.bcast(start, root=0)
    end = mpi_rec.comm.bcast(end, root=0)
    return angle_deg, start, end


def find_angle(rec, axis=0):
    rec_diff = rec.copy()
    # take different between layers to measure angle
    rec_diff[rec_diff<0] = 0
    # smooth data between IC layers and take difference
    for i in range(rec_diff.shape[1]):
        filters.gaussian_filter(rec_diff[:,i], (9, 9), output=rec_diff[:,i])
        if i > 0:
            rec_diff[:,i-1] -= rec_diff[:,i]

    # now collapse axis to correct angle (already dist on zero)
    layer_diff = np.sum(rec_diff, axis=axis)
    if axis == 2:
        layer_diff = np.rot90(layer_diff)
    
    # smooth out layer_diff
    filters.gaussian_filter(layer_diff, (3,3), output=layer_diff)
    # remove sides
    side_trim = layer_diff.shape[0]//4
    layer_diff = layer_diff[side_trim:-side_trim]
    
#     if DEBUG:
    dxchange.write_tiff(layer_diff, fname='layer_diff_%d.tiff'%axis, overwrite=True)
    angle_deg = find_angle_from_layer_diff(layer_diff, axis=1) #FIXME: axis
    return angle_deg
    
    
def find_peaks(line, peak_template=None):
    mad_val = mad(line, center=0)
    # remove beginning and ending small values
    for i in range(line.shape[0]):
        if abs(line[i]) < mad_val * 0.5:
            line[i] = 0
        else:
            break
    for i in range(line.shape[0]-1, 0, -1):
        if abs(line[i]) < mad_val * 0.5:
            line[i] = 0
        else:
            break

    if peak_template is None:
        max_peaks = argrelmax(line, order=1)[0]
        min_peaks = argrelmin(line, order=1)[0]
        peaks = np.sort(np.concatenate((max_peaks, min_peaks)))
    else:
        # match to peak_template to define peaks...
        peaks = np.cumsum(peak_template)
        peaks -= peaks[0]
        max_match = 0
        max_i = 0
        for i in range(line.shape[0] - peaks[-1]):
            #troughs = (peaks[:-1] + peaks[1:]) // 2 + i
            peak_vals = np.take(line, peaks + i)
            peak_vals[::2] *= -1
            match = abs(np.sum(peak_vals))
            #match = abs(np.sum(np.take(line, peaks + i)))
            if match > max_match:
                max_match = match
                max_i = i
        peaks += max_i
    # insert extra peak at beginning for contacts and end for aluminum
    if peaks.shape[0] > 2:
        #TODO: add peaks in reverse!
        contacts_peak = peaks[0] - (peaks[1] - peaks[0])
        if contacts_peak >= 0:
            logger.info("adding contact peak at %d"%contacts_peak)
            peaks = np.insert(peaks, 0, contacts_peak)
        aluminum_peak = peaks[-1] + 2*(peaks[-1] - peaks[-2])
        if aluminum_peak < line.shape[0]:
            logger.info("adding aluminum peak at %d"%aluminum_peak)
            peaks = np.insert(peaks, peaks.shape[0], aluminum_peak)

    return peaks


def extract_layers_mpi(mpi_rec, peak_template=None):
    # detect and extract layers from reconstruction
    # NOTE: IC should already be aligned to reconstruction grid

    # smooth along layer of IC and take difference between voxels
    # max difference should be the boundary between layers
    line = np.zeros((mpi_rec.shape[1] - 1,), np.float32)
    rec = mpi_rec.scatter(0, 5)  # add padding
    # smooth data between IC layers and take difference
    layer = None
    # average_intensity = None
    for i in range(rec.shape[1]):
        prev_layer = layer
        layer = np.copy(rec[:, i, :])
        layer[layer < 0] = 0
        layer = filters.gaussian_filter(layer, (5, 5))
        # remove differences in overall intensity
        # if average_intensity is None:
        #    average_intensity = np.mean(layer)
        # layer *= average_intensity / np.mean(layer)
        if prev_layer is not None:
            # remove padding from layers before taking diff
            offset_padding = mpi_rec.unpadded_offset - mpi_rec.offset
            size_padding = mpi_rec.size - mpi_rec.unpadded_size - offset_padding
            slc = np.s_[offset_padding: -size_padding]
            line[i - 1] = np.sum(layer[slc] - prev_layer[slc])
    filters.gaussian_filter(line, 2, output=line)
    rec = mpi_rec.scatter(0)  # remove padding
    # sum line for all nodes
    total = np.zeros_like(line)
    mpi_rec.comm.Reduce([line, MPI.FLOAT], [total, MPI.FLOAT], op=MPI.SUM, root=0)
    if mpi_rec.mpi_rank == 0:
        # calculate peaks and then share to all nodes
        trim = 10
        total[0: trim] = 0
        total[-trim:] = 0
        peaks = find_peaks(total, peak_template)
        # if DEBUG:
        #     plt.figure()
        #     plt.plot(total)
        #     plt.scatter(peaks, np.take(total, peaks))
        #     plt.savefig("line.tiff")
        logger.info("peaks from %d layers detected: %s" % (len(peaks), str(peaks)))
        logger.info("peaks spacing: %s" % (str(peaks[1:] - peaks[:-1])))
    else:
        peaks = None
    peaks = mpi_rec.comm.bcast(peaks, root=0)

    layers = np.zeros((rec.shape[0], peaks.shape[0] - 1, rec.shape[2]), dtype=rec.dtype)
    for p in reversed(range(peaks.shape[0] - 1)):  # reversed, since highest metal is first
        if peaks[p + 1] - peaks[p] >= 3:
            layers[:, p] = np.mean(rec[:, peaks[p] + 1:peaks[p + 1] - 1], axis=1)
        else:
            layers[:, p] = rec[:, (peaks[p] + peaks[p + 1]) // 2]
    mpi_layers = MpiArray(layers, axis=0)
    return mpi_layers


def write_layers_to_file_mpi(mpi_layers, path="layers"):
    layers = mpi_layers.scattermovezero(1)
    layer_names = ["contacts"]
    for i in range(1, 100):
        layer_names.append("metal%02d" % i)
        layer_names.append("via%02d-%02d" % (i, i + 1))
    # layer_names.reverse()
    # make dirs if necessary,
    if mpi_layers.mpi_rank == 0:
        os.makedirs(str(path), exist_ok=True)
    mpi_layers.comm.Barrier()
    for i in range(layers.shape[0]):
        l = mpi_layers.shape[0] - 1 - (i + mpi_layers.offset)
        dxchange.write_tiff(layers[i], fname='%s/%02d-%s' % (str(path), l, layer_names[l]), overwrite=True)


def apply_phi_correction(projs, thetas, phi_deg, out=None):
    if out is None:
        out = np.copy(projs)
    for p in range(projs.shape[0]):
        theta = thetas[p]
        proj = projs[p]
        out_proj = out[p]
        for r in range(proj.shape[0]):  # shift each row
            r_offset = r - proj.shape[0] // 2
            x_shift = -r_offset * np.tan(np.deg2rad(phi_deg)) * np.sin(theta)
            shift(proj[r], x_shift, out_proj[r], mode='nearest')
    return out
