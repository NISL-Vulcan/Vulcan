import os
from typing import Union

import torch
from torch import Tensor
from torch import distributed as dist
from torch import nn


def setup_ddp() -> int:
    """
    初始化分布式训练环境，并返回当前 GPU id。
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group('nccl', init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
    else:
        gpu = 0
    return gpu


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: Tensor) -> Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

