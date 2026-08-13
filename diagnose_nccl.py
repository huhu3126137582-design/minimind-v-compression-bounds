from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    torch.cuda.synchronize(local_rank)
    print(json.dumps({"rank": rank, "all_reduce": value.item()}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
