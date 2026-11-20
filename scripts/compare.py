"""Collect every results/*.json into one table and draw the main figure.

    python scripts/compare.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from _common import ROOT  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    args = p.parse_args()
    res_dir = Path(args.results_dir)

    rows, per_output = [], []
    for f in sorted(res_dir.glob("*.json")):
        r = json.loads(f.read_text())
        if "test" not in r:
            continue
        base = {"model": r["model"], "species": r["species"], "n_train": r["n_train"],
                "n_test": r["n_test"], "n_features": r["n_features"], "alpha": r["selection"]["alpha"]}
        rows.append({**base, **{f"mean_{k}": v for k, v in r["test"]["mean"].items()}})
        for o in r["test"]["per_output"]:
            per_output.append({**base, **o})
    if not rows:
        raise SystemExit(f"No result files in {res_dir}")

    table = pd.DataFrame(rows).sort_values(["species", "n_train", "mean_spearman"], ascending=[True, True, False])
    table.to_csv(res_dir / "summary.csv", index=False)
    pd.DataFrame(per_output).to_csv(res_dir / "summary_per_output.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Figure: per-output Spearman for each model, one panel per species/training size.
    po = pd.DataFrame(per_output)
    groups = list(po.groupby(["species", "n_train"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4), squeeze=False)
    for ax, ((species, n_train), g) in zip(axes[0], groups):
        wide = g.pivot_table(index="output", columns="model", values="spearman")
        order = wide.mean().sort_values(ascending=False).index
        wide[order].plot(kind="bar", ax=ax, width=0.8, rot=0)
        ax.set_title(f"{species.replace('_', ' ')} (n_train = {n_train})")
        ax.set_xlabel("Expression output (tissue / condition)")
        ax.set_ylabel("Test Spearman rho")
        ax.axhline(0, color="grey", lw=0.8)
        ax.legend(title="Features", fontsize=8)
    fig.tight_layout()
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "main.png", dpi=150)
    print(f"\nWrote {res_dir / 'summary.csv'} and {fig_dir / 'main.png'}")


if __name__ == "__main__":
    main()
