from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault(
    "HF_DATASETS_CACHE", str(ROOT / ".cache/huggingface/datasets")
)
sys.path.insert(0, str(ROOT / "upstream/minimind-v"))

from model.model_vlm import MiniMindVLM, VLMConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from minimind_v_bound.data.caption_dataset import (  # noqa: E402
    LockedTrainCaptionDataset,
    locked_caption_collate,
)
from minimind_v_bound.models.intrinsic_projector import (  # noqa: E402
    prepare_intrinsic_projector,
    set_projector_training_mode,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def coordinate_hash(coordinate: torch.Tensor) -> str:
    return hashlib.sha256(
        coordinate.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError(f"this smoke test requires two ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(20260813)

    config = VLMConfig(
        hidden_size=768,
        num_hidden_layers=8,
        max_seq_len=450,
        use_moe=False,
        dropout=0.0,
    )
    model = MiniMindVLM(
        config, vision_model_path=str(ROOT / "model/siglip2-base-p32-256-ve")
    )
    weights = torch.load(
        ROOT / "model/llm_768.pth", map_location="cpu", weights_only=True
    )
    incompatible = model.load_state_dict(weights, strict=False)
    missing_other = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(("vision_encoder.", "vision_proj."))
    ]
    if missing_other or incompatible.unexpected_keys:
        raise RuntimeError("base checkpoint does not match the pinned architecture")
    wrapper = prepare_intrinsic_projector(
        model, intrinsic_dimension=4096, seed=137
    )
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "upstream/minimind-v/model")
    dataset = LockedTrainCaptionDataset(
        parquet_path=ROOT / "dataset/raw/pretrain_i2t.parquet",
        cluster_database_path=ROOT / "dataset/clusters/image_clusters.sqlite3",
        train_manifest_path=ROOT / "dataset/manifests/train_clusters.jsonl",
        tokenizer=tokenizer,
        image_processor=model.processor,
        index_directory=ROOT / "dataset/manifests/train_index",
        smoke_limit=32,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        sampler=sampler,
        num_workers=0,
        collate_fn=locked_caption_collate,
    )
    model.to(device)
    set_projector_training_mode(model)
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        # Every rank reconstructs the frozen base from the same hashed files and
        # the only trainable coordinate starts at exact zeros.  Avoid broadcasting
        # the entire frozen VLM during DDP construction; only u gradients need
        # communication.
        init_sync=False,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.AdamW(
        [wrapper.subspace_params],
        lr=4e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    pixel_values = {
        key: value.to(device) for key, value in batch["pixel_values"].items()
    }
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = ddp_model(
            input_ids=input_ids, labels=labels, pixel_values=pixel_values
        )
        loss = output.loss + output.aux_loss
    loss.backward()
    gradient_norm = wrapper.subspace_params.grad.norm()
    frozen_gradient_count = sum(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name != "vision_proj.subspace_params"
    )
    if frozen_gradient_count:
        raise RuntimeError("a frozen parameter received a gradient")
    optimizer.step()

    coordinate = wrapper.subspace_params.detach()
    gathered = [torch.empty_like(coordinate) for _ in range(world_size)]
    dist.all_gather(gathered, coordinate)
    if not all(torch.equal(gathered[0], item) for item in gathered[1:]):
        raise RuntimeError("DDP ranks produced different intrinsic coordinates")
    local_summary = {
        "coordinate_sha256": coordinate_hash(coordinate),
        "gradient_norm": float(gradient_norm.cpu()),
        "loss": float(loss.detach().cpu()),
        "rank": rank,
        "row_index": int(batch["row_index"][0]),
        "per_rank_batch_size": int(input_ids.shape[0]),
        "frozen_parameter_gradients": frozen_gradient_count,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(local_rank),
        "valid_label_tokens": int(batch["valid_label_count"][0]),
        "vision_encoder_eval": not model.vision_encoder.training,
    }
    summaries: list[dict | None] = [None] * world_size
    dist.all_gather_object(summaries, local_summary)
    if rank == 0:
        print(json.dumps({
            "coordinates_identical": True,
            "trainable_parameters": 4096,
            "vision_encoder_eval_all_ranks": all(
                summary["vision_encoder_eval"] for summary in summaries
            ),
            "world_size": world_size,
            "ranks": summaries,
        }, indent=2, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
