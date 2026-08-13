from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from minimind_v_bound.certificate.sampling import (
    read_cluster_hashes,
    sample_uniform_with_replacement,
    sampling_commitment_sha256,
)


def test_sampling_is_deterministic_bounded_and_keeps_duplicates() -> None:
    seed = bytes(range(32))
    first = sample_uniform_with_replacement(
        seed=seed, population_size=7, sample_size=20
    )
    second = sample_uniform_with_replacement(
        seed=seed, population_size=7, sample_size=20
    )

    assert first.dtype == np.uint32
    assert np.array_equal(first, second)
    assert first.tolist() == [0, 1, 4, 2, 5, 2, 3, 0, 1, 1, 3, 5, 1, 5, 6, 6, 4, 3, 3, 0]
    assert int(first.min()) >= 0
    assert int(first.max()) < 7
    assert np.unique(first).size < first.size


@pytest.mark.parametrize(
    ("seed", "population_size", "sample_size"),
    [(b"short", 10, 10), (bytes(32), 0, 10), (bytes(32), 10, 0)],
)
def test_sampling_rejects_invalid_contract(
    seed: bytes, population_size: int, sample_size: int
) -> None:
    with pytest.raises(ValueError):
        sample_uniform_with_replacement(
            seed=seed, population_size=population_size, sample_size=sample_size
        )


def test_sampling_commitment_binds_seed_descriptor_and_indices() -> None:
    seed = bytes(range(32))
    indices = sample_uniform_with_replacement(
        seed=seed, population_size=17, sample_size=8
    )
    baseline = sampling_commitment_sha256(
        descriptor={"population_size": 17}, seed=seed, indices=indices
    )
    changed_indices = indices.copy()
    changed_indices[0] = (changed_indices[0] + 1) % 17

    assert baseline != sampling_commitment_sha256(
        descriptor={"population_size": 18}, seed=seed, indices=indices
    )
    assert baseline != sampling_commitment_sha256(
        descriptor={"population_size": 17}, seed=bytes(reversed(seed)), indices=indices
    )
    assert baseline != sampling_commitment_sha256(
        descriptor={"population_size": 17}, seed=seed, indices=changed_indices
    )


def test_cluster_manifest_parser_preserves_order_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clusters.jsonl"
    first = "01" * 32
    second = "ab" * 32
    path.write_text(
        json.dumps({"cluster_sha256": first})
        + "\n"
        + json.dumps({"cluster_sha256": second})
        + "\n",
        encoding="utf-8",
    )
    assert read_cluster_hashes(path) == [first, second]
    path.write_text(
        json.dumps({"cluster_sha256": first})
        + "\n"
        + json.dumps({"cluster_sha256": first})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_cluster_hashes(path)
