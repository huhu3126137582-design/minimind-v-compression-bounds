from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .checkpoint import optimizer_to_device


V6_CHECKPOINT_SCHEMA_VERSION = 1


def atomic_save_v6(payload: dict[str, Any], path: Path) -> None:
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


def build_v6_checkpoint(
    *,
    subspace_params: nn.Parameter,
    quantization_centers: nn.Parameter,
    qat_initialized: bool,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    next_batch_in_epoch: int,
    global_step: int,
    total_steps: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": V6_CHECKPOINT_SCHEMA_VERSION,
        "subspace_params": subspace_params.detach().float().cpu().clone(),
        "quantization_centers": quantization_centers.detach().float().cpu().clone(),
        "qat_initialized": bool(qat_initialized),
        "optimizer": optimizer.state_dict(),
        "progress": {
            "epoch": int(epoch),
            "next_batch_in_epoch": int(next_batch_in_epoch),
            "global_step": int(global_step),
            "total_steps": int(total_steps),
        },
        "contract": contract,
    }


def load_v6_checkpoint(
    path: Path,
    *,
    wrapper,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    expected_contract: dict[str, Any],
) -> dict[str, int | bool]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != V6_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported v6 checkpoint schema")
    if checkpoint.get("contract") != expected_contract:
        raise RuntimeError("v6 checkpoint contract does not match this run")
    coordinate = checkpoint.get("subspace_params")
    centers = checkpoint.get("quantization_centers")
    if (
        not isinstance(coordinate, torch.Tensor)
        or coordinate.dtype != torch.float32
        or coordinate.shape != wrapper.subspace_params.shape
        or not torch.isfinite(coordinate).all()
    ):
        raise RuntimeError("v6 checkpoint coordinate is invalid")
    if (
        not isinstance(centers, torch.Tensor)
        or centers.dtype != torch.float32
        or centers.shape != wrapper.quantization_centers.shape
        or not torch.isfinite(centers).all()
    ):
        raise RuntimeError("v6 checkpoint centers are invalid")
    qat_initialized = checkpoint.get("qat_initialized")
    if not isinstance(qat_initialized, bool):
        raise RuntimeError("v6 checkpoint QAT state is invalid")
    with torch.no_grad():
        wrapper.subspace_params.copy_(coordinate.to(device))
        wrapper.quantization_centers.copy_(centers.to(device))
        wrapper.qat_initialized.fill_(qat_initialized)
    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer_to_device(optimizer, device)
    progress = checkpoint["progress"]
    return {
        "epoch": int(progress["epoch"]),
        "next_batch_in_epoch": int(progress["next_batch_in_epoch"]),
        "global_step": int(progress["global_step"]),
        "total_steps": int(progress["total_steps"]),
        "qat_initialized": qat_initialized,
    }
