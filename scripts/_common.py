"""Shared helpers for the command-line scripts."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def common_args(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--species", default="arabidopsis_thaliana",
                   help="PGB gene_exp species (default: arabidopsis_thaliana, the smallest task)")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    p.add_argument("--max-train", type=int, default=None,
                   help="Subsample the training split (same subset for every model). Default: all.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    return p


def run_tag(species: str, max_train) -> str:
    return f"{species}_train{'all' if max_train is None else max_train}"
