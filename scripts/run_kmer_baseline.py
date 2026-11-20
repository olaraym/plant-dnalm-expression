"""k-mer frequencies + ridge regression baseline.

    python scripts/run_kmer_baseline.py --species arabidopsis_thaliana --k 6
"""

import time
from pathlib import Path

from _common import common_args, run_tag

import evaluate
import kmers
import pgb


def main():
    p = common_args(__doc__)
    p.add_argument("--k", type=int, nargs="+", default=[6], help="one or more k values, e.g. --k 4 5 6")
    args = p.parse_args()

    t0 = time.time()
    task = pgb.load_task(args.species, args.data_dir, args.max_train, args.seed)
    features = {split: kmers.kmer_matrix(data.seqs, tuple(args.k)) for split, data in task.items()}
    result, y_pred = evaluate.evaluate_features(features, task)
    result.update({"model": f"kmer_k{'-'.join(map(str, args.k))}", "species": args.species,
                   "max_train": args.max_train, "seed": args.seed, "seconds": round(time.time() - t0, 1)})

    out = Path(args.results_dir) / f"{result['model']}__{run_tag(args.species, args.max_train)}"
    evaluate.save_result(result, y_pred, task["test"].ids, out)
    m = result["test"]["mean"]
    print(f"{result['model']}: test mean Pearson {m['pearson']:.3f}, Spearman {m['spearman']:.3f}, "
          f"R2 {m['r2']:.3f} (alpha {result['selection']['alpha']:g}) -> {out}.json")


if __name__ == "__main__":
    main()
