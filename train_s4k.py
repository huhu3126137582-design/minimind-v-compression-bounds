from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".cache/huggingface/datasets"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
from minimind_v_bound.training.checkpoint import (  # noqa: E402
    atomic_torch_save,
    build_checkpoint,
    load_checkpoint,
    sha256_file,
)
from minimind_v_bound.training.schedule import minimind_cosine_learning_rate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal preregistered S-4K training")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per-rank-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Development-only early stop; never use for the formal two-epoch run.",
    )
    return parser.parse_args()


def initialize_distributed() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ:
        raise RuntimeError("launch with torchrun; formal training requires two ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"preregistered world size is 2, got {world_size}")
    return rank, world_size, device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def checkpoint_contract(
    *, args: argparse.Namespace, world_size: int, dataset_size: int, steps_per_epoch: int
) -> dict[str, Any]:
    return {
        "experiment_registry_sha256": sha256_file(
            ROOT / "configs/experiment_registry.yaml"
        ),
        "frozen_manifest_sha256": sha256_file(ROOT / "configs/frozen_manifest.json"),
        "implementation_contract_sha256": sha256_file(
            ROOT / "configs/implementation_contract.json"
        ),
        "train_manifest_sha256": sha256_file(
            ROOT / "dataset/manifests/train_clusters.jsonl"
        ),
        "train_index_metadata_sha256": sha256_file(
            ROOT / "dataset/manifests/train_index/metadata.json"
        ),
        "dataset_size": dataset_size,
        "epochs": args.epochs,
        "per_rank_batch_size": args.per_rank_batch_size,
        "world_size": world_size,
        "global_batch_size": args.per_rank_batch_size * world_size,
        "drop_last": True,
        "steps_per_epoch": steps_per_epoch,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "optimizer": "AdamW",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "scheduler": "minimind_cosine_v1",
        "dtype": "bfloat16",
        "training_seed": 20260815,
        "sampler_seed": 20260815,
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "intrinsic_dimension": 4096,
        "subspace_seed": 137,
    }


def save_training_checkpoint(
    *,
    rank: int,
    output_path: Path,
    wrapper,
    optimizer,
    epoch: int,
    next_batch_in_epoch: int,
    global_step: int,
    total_steps: int,
    contract: dict[str, Any],
) -> None:
    dist.barrier()
    if rank == 0:
        payload = build_checkpoint(
            subspace_params=wrapper.subspace_params,
            optimizer=optimizer,
            epoch=epoch,
            next_batch_in_epoch=next_batch_in_epoch,
            global_step=global_step,
            total_steps=total_steps,
            contract=contract,
        )
        atomic_torch_save(payload, output_path)
    dist.barrier()


def main() -> None:
    args = parse_args()
    rank, world_size, device = initialize_distributed()
    if args.epochs != 2:
        raise ValueError("the preregistered training run uses exactly 2 epochs")
    if args.per_rank_batch_size != 16:
        raise ValueError("the preregistered per-rank batch size is exactly 16")
    if args.resume is None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to start a new run in a non-empty output directory")
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # All ranks must reconstruct bitwise-identical frozen objects before DDP.
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
        raise RuntimeError("pinned LLM checkpoint does not match the architecture")
    wrapper = prepare_intrinsic_projector(model, intrinsic_dimension=4096, seed=137)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "upstream/minimind-v/model")
    dataset = LockedTrainCaptionDataset(
        parquet_path=ROOT / "dataset/raw/pretrain_i2t.parquet",
        cluster_database_path=ROOT / "dataset/clusters/image_clusters.sqlite3",
        train_manifest_path=ROOT / "dataset/manifests/train_clusters.jsonl",
        tokenizer=tokenizer,
        image_processor=model.processor,
        prompt="Describe this image.",
        image_token_length=64,
        max_sequence_length=450,
        index_directory=ROOT / "dataset/manifests/train_index",
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=20260815,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_rank_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        collate_fn=locked_caption_collate,
    )
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    contract = checkpoint_contract(
        args=args,
        world_size=world_size,
        dataset_size=len(dataset),
        steps_per_epoch=steps_per_epoch,
    )

    model.to(device)
    set_projector_training_mode(model)
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[device.index],
        broadcast_buffers=False,
        init_sync=False,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.AdamW(
        [wrapper.subspace_params],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    start_batch = 0
    global_step = 0
    if args.resume is not None:
        progress = load_checkpoint(
            args.resume,
            subspace_params=wrapper.subspace_params,
            optimizer=optimizer,
            device=device,
            expected_contract=contract,
        )
        if progress["total_steps"] != total_steps:
            raise RuntimeError("checkpoint total-step count disagrees")
        start_epoch = progress["epoch"]
        start_batch = progress["next_batch_in_epoch"]
        global_step = progress["global_step"]

    if rank == 0 and global_step == 0:
        metadata = {
            "contract": contract,
            "formal_run": args.max_steps is None,
            "max_steps": args.max_steps,
            "trainable_parameters": 4096,
            "total_steps": total_steps,
        }
        (args.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    stop_requested = False
    last_checkpoint = args.output_dir / "last.pt"
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        skip_before = start_batch if epoch == start_epoch else 0
        for batch_index, batch in enumerate(loader):
            if batch_index < skip_before:
                continue
            step_start = time.perf_counter()
            learning_rate = minimind_cosine_learning_rate(
                global_step, total_steps, args.learning_rate
            )
            optimizer.param_groups[0]["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = {
                key: value.to(device, non_blocking=True)
                for key, value in batch["pixel_values"].items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = ddp_model(
                    input_ids=input_ids,
                    labels=labels,
                    pixel_values=pixel_values,
                )
                loss = result.loss + result.aux_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at global step {global_step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [wrapper.subspace_params], args.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"non-finite gradient at global step {global_step}")
            optimizer.step()
            global_step += 1
            next_batch = batch_index + 1

            if global_step == 1:
                frozen_gradient_count = sum(
                    parameter.grad is not None
                    for name, parameter in model.named_parameters()
                    if name != "vision_proj.subspace_params"
                )
                if frozen_gradient_count:
                    raise RuntimeError("a frozen parameter received a gradient")
                gathered = [torch.empty_like(wrapper.subspace_params) for _ in range(world_size)]
                dist.all_gather(gathered, wrapper.subspace_params.detach())
                if not all(torch.equal(gathered[0], value) for value in gathered[1:]):
                    raise RuntimeError("DDP intrinsic coordinates diverged after step one")

            if rank == 0 and (
                global_step == 1 or global_step % args.log_interval == 0
            ):
                append_jsonl(
                    args.output_dir / "train.jsonl",
                    {
                        "batch_in_epoch": batch_index,
                        "coordinate_sha256": tensor_sha256(wrapper.subspace_params),
                        "epoch": epoch,
                        "global_step": global_step,
                        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                        "learning_rate": learning_rate,
                        "loss_nats_per_token_rank0": float(loss.detach().cpu()),
                        "peak_cuda_memory_bytes_rank0": torch.cuda.max_memory_allocated(),
                        "seconds": time.perf_counter() - step_start,
                    },
                )

            if global_step % args.checkpoint_interval == 0:
                save_training_checkpoint(
                    rank=rank,
                    output_path=last_checkpoint,
                    wrapper=wrapper,
                    optimizer=optimizer,
                    epoch=epoch,
                    next_batch_in_epoch=next_batch,
                    global_step=global_step,
                    total_steps=total_steps,
                    contract=contract,
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                stop_requested = True
                break

        start_batch = 0
        if stop_requested:
            save_training_checkpoint(
                rank=rank,
                output_path=last_checkpoint,
                wrapper=wrapper,
                optimizer=optimizer,
                epoch=epoch,
                next_batch_in_epoch=next_batch,
                global_step=global_step,
                total_steps=total_steps,
                contract=contract,
            )
            break
        # Epoch boundary checkpoints resume at the start of the next epoch.
        save_training_checkpoint(
            rank=rank,
            output_path=last_checkpoint,
            wrapper=wrapper,
            optimizer=optimizer,
            epoch=epoch + 1,
            next_batch_in_epoch=0,
            global_step=global_step,
            total_steps=total_steps,
            contract=contract,
        )

    if not stop_requested and global_step == total_steps:
        dist.barrier()
        if rank == 0:
            final_payload = build_checkpoint(
                subspace_params=wrapper.subspace_params,
                optimizer=optimizer,
                epoch=args.epochs,
                next_batch_in_epoch=0,
                global_step=global_step,
                total_steps=total_steps,
                contract=contract,
            )
            atomic_torch_save(final_payload, args.output_dir / "final.pt")
        dist.barrier()
    if rank == 0:
        print(json.dumps({
            "coordinate_sha256": tensor_sha256(wrapper.subspace_params),
            "global_step": global_step,
            "output_dir": str(args.output_dir),
            "stopped_early": stop_requested,
            "total_steps": total_steps,
        }, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
