from mpi4py import MPI
import logging
from . import utils
import sys
import traceback
import os
import dxchange
import shutil
import numpy as np
import collections
import tomopy

# MPI specific code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

logger = logging.getLogger(__name__)

def configure_logging(logger):
    if rank == 0:
        utils.configure_logging()
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)


def abort_mpi_on_error(exctype, value, tb):
    # log error and then properly shut down MPI
    logging.critical(''.join(traceback.format_tb(tb)))
    logging.critical('{0}: {1}'.format(exctype, value))
    MPI.COMM_WORLD.Abort(1)


def configure_abort(func=abort_mpi_on_error):
    sys.excepthook = func


# Code snippet from Chainer codebase
# https://github.com/chainer/chainer/blob/master/chainermn/communicators/_communication_utility.py
def init_ranks(mpi_comm):
    """Returns rank information of the local process in `mpi_comm`.
    Args:
        mpi_comm (type:TODO)
                 MPI Communicator from mpi4py
    Returns:
        rank_info (list):
            Elements are:
                * rank (`mpi_comm.rank`)
                * intra_rank (rank within the local computing node)
                * intra_size (number of processes on the node)
                * inter_rank (rank of the node)
                * inter_size (number of computing nodes)
    """

    global_names = mpi_comm.gather(MPI.Get_processor_name())

    if mpi_comm.rank == 0:
        name_to_global_ranks = collections.defaultdict(list)
        for global_rank, name in enumerate(global_names):
            name_to_global_ranks[name].append(global_rank)

        for global_ranks in name_to_global_ranks.values():
            global_ranks.sort()

        inter_names = sorted(
            set(global_names), key=lambda name: name_to_global_ranks[name])
        name_to_inter_rank = {
            name: inter_rank
            for inter_rank, name in enumerate(inter_names)
        }
        inter_size = len(inter_names)

        all_ranks = []
        for global_rank, name in enumerate(global_names):
            ranks = name_to_global_ranks[name]
            intra_rank = ranks.index(global_rank)
            intra_size = len(ranks)
            inter_rank = name_to_inter_rank[name]
            all_ranks.append((
                global_rank, intra_rank, intra_size,
                inter_rank, inter_size))
        my_ranks = mpi_comm.scatter(all_ranks)
    else:
        my_ranks = mpi_comm.scatter(None)

    assert my_ranks[0] == mpi_comm.rank
    return my_ranks


def write_stack_mpi(path, mpi_data):
    logger.info("writing to %s"%path)
    if mpi_data.mpi_rank == 0:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    projs = mpi_data.scatter(0)
    mpi_data.comm.Barrier()
    dxchange.writer.write_tiff_stack(projs, f"{path}/img", start=mpi_data.offset, overwrite="True")


def create_sinos_mpi(mpi_projs, ncore=None):
    # create sinograms for center finding
    mpi_sinos = mpi_projs.copy(deep=True)
    sinos = mpi_sinos.scattermovezero(1)
    np.clip(sinos, 1E-6, 1 - 1E-6, sinos)
    tomopy.minus_log(sinos, ncore, sinos)
    return mpi_sinos
