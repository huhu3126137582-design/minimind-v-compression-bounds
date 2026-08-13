from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import numpy as np


SAMPLING_ALGORITHM_VERSION = "hmac-sha256-counter-u64be-rejection-v1"
SAMPLING_DOMAIN = b"minimind-v-bound-certificate-sample-v1"


def sample_uniform_with_replacement(
    *, seed: bytes, population_size: int, sample_size: int
) -> np.ndarray:
    """Draw exact-uniform zero-based indices using a reproducible PRF stream.

    Each HMAC-SHA256 digest supplies four unsigned 64-bit big-endian words.
    Rejection before reduction removes modulo bias. Repeated indices are kept.
    """
    if len(seed) != 32:
        raise ValueError("the formal certificate seed must contain exactly 256 bits")
    if population_size <= 0 or population_size > np.iinfo(np.uint32).max:
        raise ValueError("population_size must fit a positive uint32 index space")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    modulus = 1 << 64
    acceptance_limit = modulus - (modulus % population_size)
    result = np.empty(sample_size, dtype=np.uint32)
    accepted = 0
    counter = 0
    while accepted < sample_size:
        message = SAMPLING_DOMAIN + counter.to_bytes(8, "big")
        block = hmac.new(seed, message, hashlib.sha256).digest()
        counter += 1
        for offset in range(0, len(block), 8):
            word = int.from_bytes(block[offset : offset + 8], "big")
            if word >= acceptance_limit:
                continue
            result[accepted] = word % population_size
            accepted += 1
            if accepted == sample_size:
                break
    return result


def read_cluster_hashes(path: Path) -> list[str]:
    hashes: list[str] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            value = record.get("cluster_sha256")
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"invalid cluster hash at {path}:{line_number}")
            try:
                bytes.fromhex(value)
            except ValueError as error:
                raise ValueError(f"non-hex cluster hash at {path}:{line_number}") from error
            hashes.append(value)
    if not hashes:
        raise ValueError("cluster manifest is empty")
    if len(hashes) != len(set(hashes)):
        raise ValueError("cluster manifest contains duplicate clusters")
    return hashes


def sampling_commitment_sha256(
    *, descriptor: dict[str, Any], seed: bytes, indices: np.ndarray
) -> str:
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype != np.uint32:
        raise ValueError("sample indices must be a one-dimensional uint32 array")
    digest = hashlib.sha256()
    for payload in (
        b"minimind-v-bound-certificate-sampling-commitment-v1",
        json.dumps(
            descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
        seed,
        indices.astype("<u4", copy=False).tobytes(order="C"),
    ):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()

