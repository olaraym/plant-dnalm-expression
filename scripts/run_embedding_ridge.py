"""Fit the same ridge head on saved language-model embeddings (CPU is fine).

    python scripts/run_embedding_ridge.py --species arabidopsis_thaliana --model plantcaduceus-l20 --max-train 5000
"""

import time
from pathlib import Path

import numpy as np

from _common import ROOT, common_args, run_tag

import evaluate
import pgb


def main():
    p = common_args(__doc__)
    p.add_argument("--model", default="plantcaduceus-l20")
    p.add_argument("--emb-dir", default=str(ROOT / "data" / "embeddings"))
    args = p.parse_args()

    t0 = time.time()
    task = pgb.load_task(args.species, args.data_dir, args.max_train, args.seed)
    tag = run_tag(args.species, args.max_train)
    emb_dir = Path(args.emb_dir) / args.model
    features = {}
    for split, data in task.items():
        ids = (emb_dir / f"{tag}_{split}_ids.txt").read_text().split()
        if ids != data.ids:
            raise ValueError(f"{split}: embedding ids do not match the task (different --max-train or --seed?)")
        features[split] = np.load(emb_dir / f"{tag}_{split}.npy")

    result, y_pred = evaluate.evaluate_features(features, task)
    result.update({"model": args.model, "species": args.species, "max_train": args.max_train,
                   "seed": args.seed, "seconds": round(time.time() - t0, 1)})
    out = Path(args.results_dir) / f"{args.model}__{tag}"
    evaluate.save_result(result, y_pred, task["test"].ids, out)
    m = result["test"]["mean"]
    print(f"{args.model}: test mean Pearson {m['pearson']:.3f}, Spearman {m['spearman']:.3f}, "
          f"R2 {m['r2']:.3f} (alpha {result['selection']['alpha']:g}) -> {out}.json")


if __name__ == "__main__":
    main()
