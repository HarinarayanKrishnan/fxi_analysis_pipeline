# Utility functions for tomography

import logging
import numpy as np
import os
import shutil
import psutil
import concurrent.futures as cf
from scipy.ndimage.interpolation import zoom
import dxchange


logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
                        level=logging.INFO)


def zoom_array(proj, zoom_level=0.5, ncore=None, **kwargs):
    if not ncore:
        ncore = psutil.cpu_count(True)
    output = np.empty(
        (proj.shape[0], int(round(proj.shape[1] * zoom_level)), int(round(proj.shape[2] * zoom_level))),
        dtype=proj.dtype)
    mulargs = zip(proj, [zoom_level] * proj.shape[0], output)
    with cf.ThreadPoolExecutor(ncore) as e:
        futures = [e.submit(zoom, *args, **kwargs) for args in mulargs]
        cf.wait(futures)
    return output


def center_slices(arr, size_list=None, offset_list=None):
    # calculate center slice for each axis in list
    # returns aggregate slice
    if size_list is None:
        size_list = arr.ndim*[None]
    if offset_list is None:
        offset_list = arr.ndim*[0]
    return np.s_[tuple([center_slice(arr, s, o, a) for a, (s, o) in enumerate(zip(size_list, offset_list))])]


def center_slice(arr, size=None, offset=0, axis=0):
    # return a slice to extract the center of an array for a dimension
    if size is None:
        # set size to half of the object size
        size = arr.shape[axis] // 2
    left = (arr.shape[axis] - size)//2 + offset
    return np.s_[left:left+size]


def write_stack(path, projs):
    logger.info("writing to %s"%path)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    dxchange.writer.write_tiff_stack(projs, f"{path}/img", overwrite="True")


def load_stack(path):
    logger.info(f"loading from {path}")
    num_images = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)) and name.endswith(".tiff")])
    return dxchange.reader.read_tiff_stack(f"{path}/img_00000.tiff", range(num_images))


# theoretical transmission through uniform laminar sample using zero degree
# peak transmission as a basis.
#------------------------------------------------------------------------------
def transmission(theta, theta_offset=0.0, peak=1.0):
    theta = np.array(theta)
    return np.power(peak, 1.0/np.cos((theta + theta_offset)))


def scaled_transmission(theta, theta_offset=0.0, peak=1.0):
    return transmission(theta, theta_offset, peak) / peak
