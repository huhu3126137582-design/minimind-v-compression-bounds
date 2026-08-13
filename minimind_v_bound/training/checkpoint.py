from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


CHECKPOINT_SCHEMA_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_checkpoint(
    *,
    subspace_params: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    next_batch_in_epoch: int,
    global_step: int,
    total_steps: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "subspace_params": subspace_params.detach().float().cpu().clone(),
        "optimizer": optimizer.state_dict(),
        "progress": {
            "epoch": int(epoch),
            "next_batch_in_epoch": int(next_batch_in_epoch),
            "global_step": int(global_step),
            "total_steps": int(total_steps),
        },
        "contract": contract,
    }


def load_checkpoint(
    path: Path,
    *,
    subspace_params: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    expected_contract: dict[str, Any],
) -> dict[str, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported checkpoint schema")
    if checkpoint.get("contract") != expected_contract:
        raise RuntimeError("checkpoint contract does not match this training run")
    coordinate = checkpoint.get("subspace_params")
    if not isinstance(coordinate, torch.Tensor) or coordinate.shape != subspace_params.shape:
        raise RuntimeError("checkpoint intrinsic coordinate has the wrong shape")
    if coordinate.dtype != torch.float32 or not torch.isfinite(coordinate).all():
        raise RuntimeError("checkpoint intrinsic coordinate must be finite FP32")
    with torch.no_grad():
        subspace_params.copy_(coordinate.to(device))
    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer_to_device(optimizer, device)
    progress = checkpoint["progress"]
    return {
        "epoch": int(progress["epoch"]),
        "next_batch_in_epoch": int(progress["next_batch_in_epoch"]),
        "global_step": int(progress["global_step"]),
        "total_steps": int(progress["total_steps"]),
    }

