import os
import torch

from datetime import timedelta
from loguru import logger

# Tensor Parallelism settings
RANK = int(os.getenv("RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
PP_WORLD_SIZE = int(os.getenv("PP_WORLD_SIZE", "1"))

# CUDA memory fraction
MEMORY_FRACTION = float(os.getenv("CUDA_MEMORY_FRACTION", "1.0"))


class FakeBarrier:
    def wait(self):
        pass


class FakeGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def allreduce(self, *args, **kwargs):
        return FakeBarrier()

    def allgather(self, inputs, local_tensor, **kwargs):
        assert (
            len(inputs[0]) == len(local_tensor) == 1
        ), f"{len(inputs[0])} != {len(local_tensor)} != 1, and the FakeGroup is supposed to join on simple tensors"
        for input_ in inputs:
            input_[0].data = local_tensor[0].data
        return FakeBarrier()

    def barrier(self, *args, **kwargs):
        return FakeBarrier()

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def initialize_torch_distributed():
    if torch.cuda.is_available():
        from torch.distributed import ProcessGroupNCCL

        # Set the device id.
        assert WORLD_SIZE <= torch.cuda.device_count(), "Each process is one gpu"
        device = RANK % torch.cuda.device_count()
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device)
        backend = "nccl"
        options = ProcessGroupNCCL.Options()
        options.is_high_priority_stream = True
        options._timeout = timedelta(seconds=60)
    else:
        backend = "gloo"
        options = None

    # Return: Total_Group, TP_Group, PP_Group_0_1, PP_Group_1_0, RANK, WORLD_SIZE, TP_WORLD_SIZE, PP_WORLD_SIZE
    if WORLD_SIZE == 1:
        assert WORLD_SIZE == PP_WORLD_SIZE, f"WORLD_SIZE(which is {WORLD_SIZE}) is not equal to PP_WORLD_SIZE(which is {PP_WORLD_SIZE})"
        TP_WORLD_SIZE = WORLD_SIZE
        return FakeGroup(RANK, WORLD_SIZE), FakeGroup(RANK, TP_WORLD_SIZE), None, None, RANK, WORLD_SIZE, TP_WORLD_SIZE, PP_WORLD_SIZE
    else:
        if os.getenv("DEBUG", None) == "1":
            raise NotImplementedError

        if not torch.distributed.is_initialized():
            # Call the init process.
            torch.distributed.init_process_group(
                backend=backend,
                world_size=WORLD_SIZE,
                rank=RANK,
                timeout=timedelta(seconds=60),
                pg_options=options,
            )
            torch.distributed.barrier(group=torch.distributed.group.WORLD)

            if PP_WORLD_SIZE == 1: 
                TP_WORLD_SIZE = WORLD_SIZE
                return torch.distributed.group.WORLD, torch.distributed.group.WORLD, None, None, RANK, WORLD_SIZE, TP_WORLD_SIZE, PP_WORLD_SIZE
            elif PP_WORLD_SIZE == 2:
                assert WORLD_SIZE % PP_WORLD_SIZE == 0, f"WORLD_SIZE(which is {WORLD_SIZE}) is not divisible by PP_WORLD_SIZE(which is {PP_WORLD_SIZE})"
                TP_WORLD_SIZE = WORLD_SIZE // PP_WORLD_SIZE

                # NOTE(ningpeiyang): The following logic cannot be simplified, for detail
                # https://pytorch.org/docs/stable/distributed.html#torch.distributed.new_group
                tp_ranks0 = [i for i in range(WORLD_SIZE) if i < TP_WORLD_SIZE]
                tp_group0 = torch.distributed.new_group(
                    ranks=tp_ranks0, 
                    timeout=timedelta(seconds=60), 
                    backend=backend, 
                    pg_options=options,
                )         
                tp_ranks1 = [i for i in range(WORLD_SIZE) if i >= TP_WORLD_SIZE]
                tp_group1 = torch.distributed.new_group(
                    ranks=tp_ranks1, 
                    timeout=timedelta(seconds=60), 
                    backend=backend, 
                    pg_options=options,
                )
                tp_group = tp_group0 if RANK < TP_WORLD_SIZE else tp_group1

                # For PP=2, who need pp_group_0_1: 
                # (a) The "Master Rank"(RANK=0) in FIRST part, used to do broadcast
                # (b) All ranks in LAST part, used to receive tensors sent from broadcast
                # NOTE(ningpeiyang): all processes (even not in 'pp_group_0_1') should execute the following logic, otherwise hang will be encountered
                pp_ranks_0_1 = [rank for rank in range(WORLD_SIZE) if rank == 0 or rank >= TP_WORLD_SIZE]
                pp_group_0_1 = torch.distributed.new_group(ranks=pp_ranks_0_1, 
                                                           timeout=timedelta(seconds=60), 
                                                           backend=backend, 
                                                           pg_options=options,)

                # For PP=2, who need pp_group_1_0: 
                # (a) RANK=TP_WORLD_SIZE (in LAST part), used to do broadcast
                # (b) All ranks in FIRST part, used to receive tensors sent from broadcast
                # NOTE(ningpeiyang): all processes (even not in 'pp_group_1_0') should execute the following logic, otherwise hang will be encountered
                pp_ranks_1_0 = [rank for rank in range(WORLD_SIZE) if rank <= TP_WORLD_SIZE]
                pp_group_1_0 = torch.distributed.new_group(ranks=pp_ranks_1_0, 
                                                           timeout=timedelta(seconds=60), 
                                                           backend=backend, 
                                                           pg_options=options,)

                return torch.distributed.group.WORLD, tp_group, pp_group_0_1, pp_group_1_0, RANK, WORLD_SIZE, TP_WORLD_SIZE, PP_WORLD_SIZE
            else:
                raise f"Only Support PP_WORLD_SIZE == 2"
        else:
            raise "torch.distributed is already initialized."
        
