import os

import torch
import torch.distributed as dist

dist.init_process_group("nccl")

rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

x = torch.tensor([rank], device="cuda", dtype=torch.float32)
dist.all_reduce(x)

print(f"rank={rank}, result={x.item()}")

dist.destroy_process_group()