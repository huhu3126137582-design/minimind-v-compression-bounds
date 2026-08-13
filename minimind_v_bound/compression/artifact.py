from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .quantization import QuantizedVector


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def update_length_prefixed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def canonical_compressed_model_sha256(
    *, centers: np.ndarray, assignments: np.ndarray, descriptor: dict[str, Any]
) -> str:
    if centers.dtype != np.float16 or assignments.dtype != np.uint16:
        raise ValueError("canonical compressed model requires FP16 centers and uint16 symbols")
    digest = hashlib.sha256()
    update_length_prefixed(digest, b"minimind-v-bound-compressed-model-v1")
    update_length_prefixed(digest, canonical_json_bytes(descriptor))
    update_length_prefixed(digest, centers.astype("<f2", copy=False).tobytes(order="C"))
    update_length_prefixed(digest, assignments.astype("<u2", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def canonical_reconstructed_coordinate_sha256(coordinate: np.ndarray) -> str:
    coordinate = np.asarray(coordinate)
    if coordinate.dtype != np.float32 or coordinate.ndim != 1:
        raise ValueError("canonical coordinate must be a one-dimensional FP32 array")
    digest = hashlib.sha256()
    update_length_prefixed(digest, b"minimind-v-bound-reconstructed-coordinate-v1")
    update_length_prefixed(digest, coordinate.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def canonical_named_tensor_sha256(named_tensors: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    update_length_prefixed(digest, b"minimind-v-bound-named-fp32-tensors-v1")
    for name, tensor in named_tensors:
        array = tensor.detach().float().cpu().contiguous().numpy()
        update_length_prefixed(digest, name.encode("utf-8"))
        update_length_prefixed(digest, canonical_json_bytes(list(array.shape)))
        update_length_prefixed(digest, array.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def projector_initialization_sha256(named_tensors: list[tuple[str, torch.Tensor]]) -> str:
    """Reproduce the hash contract recorded in projector_init_manifest.json."""
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(canonical_json_bytes(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_quantized_arrays(output_directory: Path, result: QuantizedVector) -> dict[str, str]:
    files = {
        "centers": output_directory / "centers.npy",
        "assignments": output_directory / "assignments.npy",
        "counts": output_directory / "counts.npy",
        "reconstructed_u": output_directory / "reconstructed_u.npy",
    }
    np.save(files["centers"], result.centers, allow_pickle=False)
    np.save(files["assignments"], result.assignments, allow_pickle=False)
    np.save(files["counts"], result.counts, allow_pickle=False)
    np.save(files["reconstructed_u"], result.reconstructed, allow_pickle=False)
    return {f"{name}_sha256": sha256_file(path) for name, path in files.items()}


def load_and_verify_quantized_arrays(output_directory: Path) -> tuple[dict[str, Any], QuantizedVector]:
    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    paths = {
        "centers": output_directory / "centers.npy",
        "assignments": output_directory / "assignments.npy",
        "counts": output_directory / "counts.npy",
        "reconstructed_u": output_directory / "reconstructed_u.npy",
    }
    for name, path in paths.items():
        if sha256_file(path) != manifest["files"][f"{name}_sha256"]:
            raise RuntimeError(f"quantized artifact hash mismatch: {name}")
    result = QuantizedVector(
        centers=np.load(paths["centers"], allow_pickle=False),
        assignments=np.load(paths["assignments"], allow_pickle=False),
        counts=np.load(paths["counts"], allow_pickle=False),
        reconstructed=np.load(paths["reconstructed_u"], allow_pickle=False),
        initial_assignments=np.empty(0, dtype=np.uint16),
    )
    result.validate(
        dimension=manifest["quantization"]["dimension"],
        levels=manifest["quantization"]["levels"],
    )
    return manifest, result
