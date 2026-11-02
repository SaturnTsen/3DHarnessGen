import os

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    if rank == 0:
        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        dist.send(x, dst=1)

    elif rank == 1:
        x = torch.empty(3, device="cuda")
        dist.recv(x, src=0)
        print(x)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()