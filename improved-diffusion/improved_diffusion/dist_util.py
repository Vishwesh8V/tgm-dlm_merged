"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
try:
    from mpi4py import MPI
except Exception:
    class DummyCOMM_WORLD:
        @staticmethod
        def Get_rank(): return 0
        @staticmethod
        def bcast(data, root=0): return data
    class DummyMPI:
        COMM_WORLD = DummyCOMM_WORLD
    MPI = DummyMPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 1 #8

SETUP_RETRY_COUNT = 3


def setup_dist(rank, world_size, port=None):
    """
    Setup a distributed process group.

    Port selection priority:
      1. `port` argument if explicitly passed
      2. MASTER_PORT environment variable if set
      3. Auto-detected free port via _find_free_port()

    Using a free port by default means multiple training runs on the same
    machine will never collide (the old hardcoded 12145 caused
    'Address already in use' when a second job was launched).
    """
    print("IN AUG DIST setup")
    if dist.is_initialized():
        return

    if port is None:
        port = os.environ.get('MASTER_PORT', None)
    if port is None:
        port = str(_find_free_port())

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    print(f"[dist_util] Using MASTER_PORT={port}")
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
