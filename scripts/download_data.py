"""Download one species' gene-expression task from the Plants Genomic Benchmark and show its layout.

    python scripts/download_data.py --species arabidopsis_thaliana
"""

from _common import common_args

import pgb


def main():
    args = common_args(__doc__).parse_args()
    paths = pgb.download(args.species, args.data_dir)
    print("Downloaded:")
    for split, path in paths.items():
        print(f"  {split:5s} {path}")

    # Show the first headers so the label layout can be confirmed before modelling.
    print("\nFirst 3 headers of the training file:")
    with pgb._open(paths["train"]) as fh:
        shown = 0
        for line in fh:
            if line.startswith(">"):
                print("  " + line.strip()[:160])
                shown += 1
                if shown == 3:
                    break

    task = pgb.load_task(args.species, args.data_dir)
    for split, data in task.items():
        lengths = [len(s) for s in data.seqs]
        print(f"{split:5s} n={len(data):6d}  outputs={data.y.shape[1]}  "
              f"length min/median/max = {min(lengths)}/{sorted(lengths)[len(lengths)//2]}/{max(lengths)}")


if __name__ == "__main__":
    main()
