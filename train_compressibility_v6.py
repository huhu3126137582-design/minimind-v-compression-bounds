from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

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

from minimind_v_bound.compression.artifact import projector_initialization_sha256  # noqa: E402
from minimind_v_bound.configuration_v6_compressibility import (  # noqa: E402
    V6_CONFIGURATION_IDS,
    load_v6_registry,
    resolve_v6_candidate,
    sha256_file,
    verify_v6_protocol_freeze,
)
from minimind_v_bound.data.caption_dataset import (  # noqa: E402
    LockedTrainCaptionDataset,
    locked_caption_collate,
)
from minimind_v_bound.models.intrinsic_projector import set_projector_training_mode  # noqa: E402
from minimind_v_bound.models.qat_projector_v6 import (  # noqa: E402
    QAT_PROJECTOR_VERSION,
    prepare_qat_projector_v6,
)
from minimind_v_bound.models.structured_projection_v6 import (  # noqa: E402
    PROJECTION_V6_VERSION,
)
from minimind_v_bound.compression.qat_quantization_v6 import (  # noqa: E402
    QAT_QUANTIZATION_VERSION,
)
from minimind_v_bound.training.checkpoint_v6 import (  # noqa: E402
    atomic_save_v6,
    build_v6_checkpoint,
    load_v6_checkpoint,
)
from minimind_v_bound.training.schedule import minimind_cosine_learning_rate  # noqa: E402
from train_subspace_v2 import (  # noqa: E402
    append_jsonl,
    initialize_distributed,
    set_seed,
    tensor_sha256,
)


TRAINER_VERSION = "compressibility-v6-ddp-coordinate-qat-v1"
REGISTRY_DEFAULT = ROOT / "configs/experiment_registry_v6_compressibility.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one fixed v6 compressibility candidate")
    parser.add_argument("--configuration-id", choices=V6_CONFIGURATION_IDS, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--development-qat-start-step",
        type=int,
        help="Development only: exercise the QAT boundary early in a smoke run.",
    )
    return parser.parse_args()


def build_v6_training_contract(
    *,
    registry_path: Path,
    registry: dict[str, Any],
    candidate,
    world_size: int,
    dataset_size: int,
    steps_per_epoch: int,
    qat_start_step: int,
    development_override: bool,
    factor_initialization_sha256: str | None,
) -> dict[str, Any]:
    training = registry["training"]
    return {
        "schema_version": 1,
        "trainer_version": TRAINER_VERSION,
        "qat_projector_version": QAT_PROJECTOR_VERSION,
        "qat_quantization_version": QAT_QUANTIZATION_VERSION,
        "projection_version": PROJECTION_V6_VERSION,
        "trainer_source_sha256": registry["frozen_inputs"][
            "training_contract_trainer_sha256"
        ],
        # Preserve compatibility with Q=11 checkpoints started under the
        # original 15-candidate registry; their training contract is unchanged.
        "registry_sha256": registry["frozen_inputs"][
            "training_contract_registry_sha256"
        ],
        "implementation_contract_sha256": sha256_file(
            ROOT / registry["frozen_inputs"]["implementation_contract"]
        ),
        "frozen_manifest_sha256": sha256_file(ROOT / "configs/frozen_manifest.json"),
        "projector_init_manifest_sha256": sha256_file(
            ROOT / "configs/projector_init_manifest.json"
        ),
        "train_manifest_sha256": sha256_file(
            ROOT / registry["frozen_inputs"]["train_manifest"]
        ),
        "train_index_metadata_sha256": sha256_file(
            ROOT / registry["frozen_inputs"]["train_index"] / "metadata.json"
        ),
        "candidate_id": candidate.candidate_id,
        "configuration_id": candidate.configuration_id,
        "parameterization": candidate.parameterization,
        "lora_rank": candidate.rank,
        "intrinsic_dimension": candidate.intrinsic_dimension,
        "quantization_levels": candidate.quantization_levels,
        "factor_initialization_sha256": factor_initialization_sha256,
        "dataset_size": dataset_size,
        "epochs": training["epochs"],
        "steps_per_epoch": steps_per_epoch,
        "total_steps": training["epochs"] * steps_per_epoch,
        "qat_start_step": qat_start_step,
        "development_qat_start_override": development_override,
        "world_size": world_size,
        "per_rank_batch_size": training["per_gpu_batch_size"],
        "global_batch_size": training["global_batch_size"],
        "learning_rate": training["learning_rate"],
        "coordinate_weight_decay": training["coordinate_weight_decay"],
        "center_weight_decay": training["center_weight_decay"],
        "training_seed": training["seed"],
        "sampler_seed": training["sampler_seed"],
        "checkpoint_selection": "final_epoch",
        "heldout_validation_accessed": False,
        "existing_certification_sample_accessed": False,
    }


def save_checkpoint(
    *,
    rank: int,
    path: Path,
    wrapper,
    optimizer,
    epoch: int,
    next_batch: int,
    global_step: int,
    total_steps: int,
    contract: dict[str, Any],
) -> None:
    dist.barrier()
    if rank == 0:
        atomic_save_v6(
            build_v6_checkpoint(
                subspace_params=wrapper.subspace_params,
                quantization_centers=wrapper.quantization_centers,
                qat_initialized=bool(wrapper.qat_initialized.item()),
                optimizer=optimizer,
                epoch=epoch,
                next_batch_in_epoch=next_batch,
                global_step=global_step,
                total_steps=total_steps,
                contract=contract,
            ),
            path,
        )
    dist.barrier()


def main() -> None:
    args = parse_args()
    registry_path = args.registry.resolve()
    registry = load_v6_registry(registry_path)
    candidate = resolve_v6_candidate(registry, args.configuration_id, root=ROOT)
    formal_output = candidate.output_directory / "training"
    output_directory = args.output_dir.resolve() if args.output_dir else formal_output
    development = args.max_steps is not None
    if development:
        if args.max_steps is None or args.max_steps <= 0:
            raise ValueError("development max_steps must be positive")
        development_root = (ROOT / "runs/v6_development").resolve()
        if args.output_dir is None or development_root not in output_directory.parents:
            raise ValueError("v6 development output must be under runs/v6_development")
        if args.development_qat_start_step is not None and not (
            0 <= args.development_qat_start_step < args.max_steps
        ):
            raise ValueError("development QAT start must lie inside the smoke run")
    else:
        if args.development_qat_start_step is not None:
            raise ValueError("formal training forbids the development QAT override")
        verify_v6_protocol_freeze(root=ROOT, registry_path=registry_path)
        if output_directory != formal_output:
            raise ValueError("formal v6 output must equal the registered directory")

    training = registry["training"]
    rank, world_size, device = initialize_distributed(training["planned_world_size"])
    if args.resume is None and output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"refusing non-empty v6 output: {output_directory}")
    if rank == 0:
        output_directory.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    set_seed(training["projector_constructor_seed"])
    config = VLMConfig(
        hidden_size=768,
        num_hidden_layers=8,
        max_seq_len=450,
        use_moe=False,
        dropout=training["dropout"],
    )
    model = MiniMindVLM(
        config, vision_model_path=str(ROOT / "model/siglip2-base-p32-256-ve")
    )
    expected_projector_hash = json.loads(
        (ROOT / "configs/projector_init_manifest.json").read_text(encoding="utf-8")
    )["canonical_state_sha256"]
    if projector_initialization_sha256(list(model.vision_proj.named_parameters())) != expected_projector_hash:
        raise RuntimeError("v6 Projector initialization differs from the frozen base")
    weights = torch.load(ROOT / "model/llm_768.pth", map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(weights, strict=False)
    missing_other = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(("vision_encoder.", "vision_proj."))
    ]
    if missing_other or incompatible.unexpected_keys:
        raise RuntimeError("pinned LLM checkpoint does not match v6 architecture")
    common = registry["candidate_family"]["common_side_information"]
    wrapper = prepare_qat_projector_v6(
        model,
        parameterization=candidate.parameterization,
        intrinsic_dimension=candidate.intrinsic_dimension,
        quantization_levels=candidate.quantization_levels,
        rank=candidate.rank,
        lora_scale=common["lora_scale"],
        subspace_seed=common["subspace_seed"],
        factor_init_seed=common["factor_init_seed"],
    )
    factor_hash = (
        wrapper.factor_initialization_sha256()
        if candidate.parameterization == "sublora"
        else None
    )
    implementation = json.loads(
        (ROOT / registry["frozen_inputs"]["implementation_contract"]).read_text(
            encoding="utf-8"
        )
    )
    if candidate.parameterization == "sublora":
        expected = implementation["sublora"][f"rank_{candidate.rank}_phi0_sha256"]
        if factor_hash != expected:
            raise RuntimeError("v6 SubLoRA factor initialization hash differs")
    set_seed(training["seed"])

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "upstream/minimind-v/model")
    dataset = LockedTrainCaptionDataset(
        parquet_path=ROOT / registry["frozen_inputs"]["raw_dataset"],
        cluster_database_path=ROOT / "dataset/clusters/image_clusters.sqlite3",
        train_manifest_path=ROOT / registry["frozen_inputs"]["train_manifest"],
        tokenizer=tokenizer,
        image_processor=model.processor,
        prompt=registry["evaluation"]["canonical_prompt"],
        image_token_length=registry["evaluation"]["image_token_length"],
        max_sequence_length=registry["evaluation"]["max_sequence_length"],
        index_directory=ROOT / registry["frozen_inputs"]["train_index"],
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=training["sampler_seed"],
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=training["per_gpu_batch_size"],
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        collate_fn=locked_caption_collate,
    )
    steps_per_epoch = len(loader)
    total_steps = training["epochs"] * steps_per_epoch
    if (
        steps_per_epoch != training["expected_steps_per_epoch"]
        or total_steps != training["expected_total_steps"]
    ):
        raise RuntimeError("v6 dataset step count differs from registry")
    formal_qat_steps = math.ceil(
        total_steps
        * registry["qat"]["final_fraction_numerator"]
        / registry["qat"]["final_fraction_denominator"]
    )
    formal_qat_start = total_steps - formal_qat_steps
    if formal_qat_start != registry["qat"]["expected_qat_start_step"]:
        raise RuntimeError("v6 QAT boundary differs from registry")
    qat_start = (
        args.development_qat_start_step
        if args.development_qat_start_step is not None
        else formal_qat_start
    )
    contract = build_v6_training_contract(
        registry_path=registry_path,
        registry=registry,
        candidate=candidate,
        world_size=world_size,
        dataset_size=len(dataset),
        steps_per_epoch=steps_per_epoch,
        qat_start_step=qat_start,
        development_override=args.development_qat_start_step is not None,
        factor_initialization_sha256=factor_hash,
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
        [
            {
                "params": [wrapper.subspace_params],
                "weight_decay": training["coordinate_weight_decay"],
            },
            {
                "params": [wrapper.quantization_centers],
                "weight_decay": training["center_weight_decay"],
            },
        ],
        lr=training["learning_rate"],
        betas=tuple(training["betas"]),
        eps=training["epsilon"],
    )
    start_epoch = start_batch = global_step = 0
    if args.resume is not None:
        progress = load_v6_checkpoint(
            args.resume.resolve(),
            wrapper=wrapper,
            optimizer=optimizer,
            device=device,
            expected_contract=contract,
        )
        start_epoch = int(progress["epoch"])
        start_batch = int(progress["next_batch_in_epoch"])
        global_step = int(progress["global_step"])
        if int(progress["total_steps"]) != total_steps:
            raise RuntimeError("v6 resumed total steps differ")
        if global_step >= qat_start and not bool(progress["qat_initialized"]):
            raise RuntimeError("v6 resumed after QAT boundary without centers")

    if rank == 0 and global_step == 0:
        (output_directory / "run_metadata.json").write_text(
            json.dumps(
                {
                    "contract": contract,
                    "formal_run": not development,
                    "max_steps": args.max_steps,
                    "total_steps": total_steps,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    stop_requested = False
    next_batch = start_batch
    last_path = output_directory / "last.pt"
    for epoch in range(start_epoch, training["epochs"]):
        sampler.set_epoch(epoch)
        skip_before = start_batch if epoch == start_epoch else 0
        for batch_index, batch in enumerate(loader):
            if batch_index < skip_before:
                continue
            if global_step >= qat_start and not bool(wrapper.qat_initialized.item()):
                wrapper.initialize_qat_centers()
                # The center has an exact-zero graph contribution before QAT so
                # DDP can keep find_unused_parameters=False. AdamW nevertheless
                # creates and advances zero-valued moments in that phase; reset
                # them here so learned-center optimization starts at the frozen
                # QAT boundary rather than inheriting a fictitious step count.
                optimizer.state.pop(wrapper.quantization_centers, None)
                gathered = [
                    torch.empty_like(wrapper.quantization_centers)
                    for _ in range(world_size)
                ]
                dist.all_gather(gathered, wrapper.quantization_centers.detach())
                if not all(torch.equal(gathered[0], item) for item in gathered[1:]):
                    raise RuntimeError("v6 QAT center initialization diverged across ranks")
            qat_active = global_step >= qat_start
            wrapper.set_qat_forward(qat_active)
            step_start = time.perf_counter()
            learning_rate = minimind_cosine_learning_rate(
                global_step, total_steps, training["learning_rate"]
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = {
                key: value.to(device, non_blocking=True)
                for key, value in batch["pixel_values"].items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = ddp_model(
                    input_ids=input_ids, labels=labels, pixel_values=pixel_values
                )
                loss = result.loss + result.aux_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite v6 loss at step {global_step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [wrapper.subspace_params, wrapper.quantization_centers],
                training["gradient_clip_norm"],
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"non-finite v6 gradient at step {global_step}")
            optimizer.step()
            global_step += 1
            next_batch = batch_index + 1

            if global_step == 1:
                frozen_gradients = sum(
                    parameter.grad is not None
                    for name, parameter in model.named_parameters()
                    if name
                    not in (
                        "vision_proj.subspace_params",
                        "vision_proj.quantization_centers",
                    )
                )
                if frozen_gradients:
                    raise RuntimeError("a frozen v6 parameter received a gradient")
            if rank == 0 and (global_step == 1 or global_step % args.log_interval == 0):
                append_jsonl(
                    output_directory / "train.jsonl",
                    {
                        "candidate_id": candidate.candidate_id,
                        "configuration_id": candidate.configuration_id,
                        "epoch": epoch,
                        "batch_in_epoch": batch_index,
                        "global_step": global_step,
                        "phase": "qat" if qat_active else "continuous",
                        "coordinate_sha256": tensor_sha256(wrapper.subspace_params),
                        "centers_sha256": tensor_sha256(wrapper.quantization_centers),
                        "loss_nats_per_token_rank0": float(loss.detach().cpu()),
                        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                        "learning_rate": learning_rate,
                        "seconds": time.perf_counter() - step_start,
                        "peak_cuda_memory_bytes_rank0": torch.cuda.max_memory_allocated(),
                    },
                )
            if global_step % args.checkpoint_interval == 0:
                save_checkpoint(
                    rank=rank,
                    path=last_path,
                    wrapper=wrapper,
                    optimizer=optimizer,
                    epoch=epoch,
                    next_batch=next_batch,
                    global_step=global_step,
                    total_steps=total_steps,
                    contract=contract,
                )
            if development and global_step >= args.max_steps:
                stop_requested = True
                break
        start_batch = 0
        if stop_requested:
            save_checkpoint(
                rank=rank,
                path=last_path,
                wrapper=wrapper,
                optimizer=optimizer,
                epoch=epoch,
                next_batch=next_batch,
                global_step=global_step,
                total_steps=total_steps,
                contract=contract,
            )
            break
        save_checkpoint(
            rank=rank,
            path=last_path,
            wrapper=wrapper,
            optimizer=optimizer,
            epoch=epoch + 1,
            next_batch=0,
            global_step=global_step,
            total_steps=total_steps,
            contract=contract,
        )

    if not stop_requested and global_step == total_steps:
        if not bool(wrapper.qat_initialized.item()):
            raise RuntimeError("formal v6 training ended without QAT")
        save_checkpoint(
            rank=rank,
            path=output_directory / "final.pt",
            wrapper=wrapper,
            optimizer=optimizer,
            epoch=training["epochs"],
            next_batch=0,
            global_step=global_step,
            total_steps=total_steps,
            contract=contract,
        )
    if rank == 0:
        print(
            json.dumps(
                {
                    "configuration_id": candidate.configuration_id,
                    "global_step": global_step,
                    "qat_initialized": bool(wrapper.qat_initialized.item()),
                    "stopped_early": stop_requested,
                    "output_directory": str(output_directory),
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
