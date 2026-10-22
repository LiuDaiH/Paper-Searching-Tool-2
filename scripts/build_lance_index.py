"""Build the LanceDB paper table from existing BGE-M3 vector shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hotspot.config import load_config  # noqa: E402
from hotspot.index.lance_store import build_lance_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", choices=("meeting", "arxiv"), default="meeting")
    args = parser.parse_args()
    manifest = build_lance_index(
        load_config(),
        out_dir=Path(args.out) if args.out else None,
        limit=args.limit,
        source=args.source,
    )
    print(
        f"lance | papers={manifest['papers']} dim={manifest['dimension']} "
        f"out={args.out or 'workspace/lance'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
