from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from minimind_v_bound.compression.artifact import (
    canonical_compressed_model_sha256,
    load_and_verify_quantized_arrays,
    write_quantized_arrays,
)
from minimind_v_bound.compression.quantization import (
    quantize_equal_width_bin_means,
)


def test_quantized_artifact_round_trip_and_hash_check(tmp_path: Path) -> None:
    result = quantize_equal_width_bin_means(
        np.linspace(-1.0, 1.0, 17, dtype=np.float32), levels=5
    )
    file_hashes = write_quantized_arrays(tmp_path, result)
    manifest = {
        "quantization": {"dimension": 17, "levels": 5},
        "files": file_hashes,
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    loaded_manifest, loaded = load_and_verify_quantized_arrays(tmp_path)
    assert loaded_manifest == manifest
    assert np.array_equal(loaded.centers, result.centers)
    assert np.array_equal(loaded.assignments, result.assignments)
    assert np.array_equal(loaded.reconstructed, result.reconstructed)

    assignments_path = tmp_path / "assignments.npy"
    payload = bytearray(assignments_path.read_bytes())
    payload[-1] ^= 1
    assignments_path.write_bytes(payload)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_and_verify_quantized_arrays(tmp_path)


def test_compressed_hash_commits_to_descriptor_and_symbols() -> None:
    result = quantize_equal_width_bin_means(
        np.linspace(-1.0, 1.0, 9, dtype=np.float32), levels=3
    )
    first = canonical_compressed_model_sha256(
        centers=result.centers,
        assignments=result.assignments,
        descriptor={"configuration_id": "S-4K"},
    )
    changed_descriptor = canonical_compressed_model_sha256(
        centers=result.centers,
        assignments=result.assignments,
        descriptor={"configuration_id": "S-8K"},
    )
    changed_symbols = result.assignments.copy()
    changed_symbols[0] = (changed_symbols[0] + 1) % 3
    changed_assignment_hash = canonical_compressed_model_sha256(
        centers=result.centers,
        assignments=changed_symbols,
        descriptor={"configuration_id": "S-4K"},
    )

    assert first != changed_descriptor
    assert first != changed_assignment_hash
