from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from minimind_v_bound.data.caption_dataset import load_training_example_index


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable train-only row index")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    row_path = args.output_directory / "row_indices.npy"
    caption_path = args.output_directory / "caption_indices.npy"
    metadata_path = args.output_directory / "metadata.json"
    if any(path.exists() for path in (row_path, caption_path, metadata_path)):
        raise FileExistsError("refusing to overwrite an existing training index")

    rows, captions = load_training_example_index(
        args.database, args.train_manifest
    )
    np.save(row_path, rows, allow_pickle=False)
    np.save(caption_path, captions, allow_pickle=False)
    metadata = {
        "schema_version": 1,
        "caption_examples": len(rows),
        "distinct_source_rows": int(np.unique(rows).size),
        "row_indices_dtype": str(rows.dtype),
        "caption_indices_dtype": str(captions.dtype),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "row_indices_sha256": sha256_file(row_path),
        "caption_indices_sha256": sha256_file(caption_path),
    }
    with metadata_path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
