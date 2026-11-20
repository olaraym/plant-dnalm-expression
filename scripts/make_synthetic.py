"""Write a small synthetic task in the PGB gene_exp file layout, for offline smoke tests and CI.

Expression of each 'tissue' is driven by counts of known plant promoter motifs plus noise, so a
working k-mer pipeline must recover a clearly positive correlation. Fully deterministic.

    python scripts/make_synthetic.py --out data/synthetic
"""

import argparse
from pathlib import Path

import numpy as np

MOTIFS = ("TATAAA", "CACGTG", "GCCGCC")  # TATA box, G-box, GCC box
SPLIT_SEEDS = {"train": 11, "valid": 12, "test": 13}


def make_split(rng, n, length, weights):
    seqs, ys = [], []
    for _ in range(n):
        s = list(rng.choice(list("ACGT"), size=length))
        for motif, copies in zip(MOTIFS, rng.poisson(2, size=len(MOTIFS))):
            for _ in range(copies):
                pos = rng.integers(0, length - len(motif))
                s[pos : pos + len(motif)] = list(motif)
        seq = "".join(s)
        observed = np.array([seq.count(m) for m in MOTIFS])
        ys.append(observed @ weights + rng.normal(0, 0.5, size=weights.shape[1]))
        seqs.append(seq)
    return seqs, np.array(ys)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--species", default="arabidopsis_thaliana")
    p.add_argument("--n", type=int, nargs=3, default=[600, 150, 150], help="train valid test sizes")
    p.add_argument("--length", type=int, default=1500)
    p.add_argument("--outputs", type=int, default=4)
    args = p.parse_args()

    weights = np.random.default_rng(1).normal(0, 1, size=(len(MOTIFS), args.outputs))
    out = Path(args.out) / "gene_exp"
    out.mkdir(parents=True, exist_ok=True)
    for split, n in zip(("train", "valid", "test"), args.n):
        seqs, ys = make_split(np.random.default_rng(SPLIT_SEEDS[split]), n, args.length, weights)
        path = out / f"{args.species}_{split}.fa"
        with open(path, "w") as fh:
            for i, (s, y) in enumerate(zip(seqs, ys)):
                fh.write(f">gene{i}_{split}|" + "|".join(f"{v:.4f}" for v in y) + "\n")
                for j in range(0, len(s), 80):  # multi-line FASTA on purpose
                    fh.write(s[j : j + 80] + "\n")
        print(f"wrote {path} ({n} sequences)")


if __name__ == "__main__":
    main()
