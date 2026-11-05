"""Pack a `train-merged` score table for git, or rebuild it (src/models/score_pack.py).

    make pack-merged-scores   SCORES=<scores.parquet> PACK_DIR=<dir>
    make unpack-merged-scores PACK_DIR=<dir> OUT=<rebuilt.parquet>

The table is about 1 GB, over GitHub's 100 MB file limit. `pack` writes parts under
50 MB each and a MANIFEST.json, drops the rank-blend columns (recomputed on rebuild)
and stores the sequence nets' scores as float32 (exact; checked). It rebuilds the table
from the parts and refuses to finish unless that equals the original. Both commands
refuse to overwrite existing output unless --overwrite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import yaml

from src.models.score_pack import MANIFEST, pack, rebuild
from src.outputs import refuse_overwrite

MODEL_CONFIG_PATH = Path("configs/model.yaml")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("scores", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument(
        "--float32",
        nargs="*",
        default=["lstm", "tcn"],
        help="columns stored as float32 (the sequence nets' scores); refused if not exact",
    )
    p.add_argument(
        "--ensemble",
        action="append",
        default=[],
        metavar="NAME=M1,M2,...",
        help="a blend to drop and recompute, members in order. Default: "
        "merged_models.ensembles in configs/model.yaml. The pack records its own copy.",
    )
    p.add_argument("--rows-per-part", type=int, default=2_800_000)
    p.add_argument("--overwrite", action="store_true")
    r = sub.add_parser("rebuild")
    r.add_argument("pack_dir", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "pack":
        refuse_overwrite([args.out_dir / MANIFEST], overwrite=args.overwrite)
        if args.ensemble:
            ensembles = {
                name: members.split(",")
                for name, members in (item.split("=", 1) for item in args.ensemble)
            }
        else:
            config = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
            ensembles = {k: list(v) for k, v in config["merged_models"]["ensembles"].items()}
        manifest = pack(
            pl.read_parquet(args.scores),
            args.out_dir,
            ensembles=ensembles,
            float32_columns=args.float32,
            rows_per_part=args.rows_per_part,
        )
        print(
            f"packed {manifest['rows']} rows into {len(manifest['parts'])} parts; "
            "rebuild verified equal to the original"
        )
    else:
        refuse_overwrite([args.out], overwrite=args.overwrite)
        rebuild(args.pack_dir).write_parquet(args.out, compression="zstd")
        print(f"rebuilt table written to {args.out}")


if __name__ == "__main__":
    main()
