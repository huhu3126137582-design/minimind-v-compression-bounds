from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

from minimind_v_bound.data.clusters import deterministic_cluster_split


def write_manifest(path: Path, hashes: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for cluster_hash in hashes:
            line = json.dumps(
                {"cluster_sha256": cluster_hash}, separators=(",", ":"), sort_keys=True
            ) + "\n"
            output.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic cluster split")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    with sqlite3.connect(f"file:{args.database}?mode=ro", uri=True) as connection:
        cluster_hashes = [row[0] for row in connection.execute(
            "SELECT image_sha256 FROM images ORDER BY image_sha256"
        )]
    train, test = deterministic_cluster_split(
        cluster_hashes, train_fraction=args.train_fraction, seed=args.seed
    )
    train_digest = write_manifest(args.train_output, train)
    test_digest = write_manifest(args.test_output, test)
    os.chmod(args.test_output, 0o400)
    print(json.dumps({
        "clusters": len(cluster_hashes),
        "train_clusters": len(train),
        "test_clusters": len(test),
        "train_manifest_sha256": train_digest,
        "test_manifest_sha256": test_digest,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
