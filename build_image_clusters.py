from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimind_v_bound.data.clusters import build_cluster_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact decoded-pixel image clusters")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--reject-invalid", action="store_true")
    args = parser.parse_args()
    result = build_cluster_database(
        args.parquet,
        args.database,
        batch_size=args.batch_size,
        reject_invalid=args.reject_invalid,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
