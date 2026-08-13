from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def read_manifest(path: Path) -> tuple[list[str], str]:
    digest = hashlib.sha256()
    hashes = []
    with path.open("rb") as source:
        for raw_line in source:
            digest.update(raw_line)
            record = json.loads(raw_line)
            hashes.append(record["cluster_sha256"])
    return hashes, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train/test cluster isolation")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    expected = json.loads(args.manifest.read_text(encoding="utf-8"))
    train, train_digest = read_manifest(args.train)
    test, test_digest = read_manifest(args.test)
    train_set, test_set = set(train), set(test)
    assert len(train) == len(train_set) == expected["train_clusters"]
    assert len(test) == len(test_set) == expected["test_clusters"]
    assert train_set.isdisjoint(test_set)
    assert train_digest == expected["train_manifest_sha256"]
    assert test_digest == expected["test_manifest_sha256"]

    with sqlite3.connect(f"file:{args.database}?mode=ro", uri=True) as connection:
        database_hashes = {row[0] for row in connection.execute(
            "SELECT image_sha256 FROM images"
        )}
        source_rows = connection.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
        rejected_rows = connection.execute(
            "SELECT COUNT(*) FROM rejected_rows"
        ).fetchone()[0]
    assert train_set | test_set == database_hashes
    assert source_rows == expected["source_rows"]
    assert rejected_rows == expected["rejected_rows"]
    print(json.dumps({
        "database_clusters": len(database_hashes),
        "intersection": 0,
        "rejected_rows": rejected_rows,
        "source_rows": source_rows,
        "test_clusters": len(test),
        "train_clusters": len(train),
        "union_matches_database": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
