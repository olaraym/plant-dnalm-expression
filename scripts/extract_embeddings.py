"""Extract frozen language-model embeddings for every split (run on a GPU, e.g. Google Colab).

    python scripts/extract_embeddings.py --species arabidopsis_thaliana --model plantcaduceus-l20 --max-train 5000

Writes data/embeddings/<model>/<run tag>_<split>.npy and matching _ids.txt files.
"""

import time
from pathlib import Path

import numpy as np

from _common import ROOT, common_args, run_tag

import embed
import pgb


def main():
    p = common_args(__doc__)
    p.add_argument("--model", default="plantcaduceus-l20", choices=sorted(embed.MODELS))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--emb-dir", default=str(ROOT / "data" / "embeddings"))
    args = p.parse_args()

    task = pgb.load_task(args.species, args.data_dir, args.max_train, args.seed)
    tokenizer, model, spec = embed.load_model(args.model, args.device)
    out_dir = Path(args.emb_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = run_tag(args.species, args.max_train)

    for split, data in task.items():
        target = out_dir / f"{tag}_{split}.npy"
        if target.exists():
            print(f"{target} exists; skipping")
            continue
        t0 = time.time()
        X = embed.embed_sequences(data.seqs, tokenizer, model, spec.window_bp, args.batch_size, args.device)
        np.save(target, X)
        (out_dir / f"{tag}_{split}_ids.txt").write_text("\n".join(data.ids) + "\n")
        print(f"{split}: {X.shape} in {time.time() - t0:.0f}s -> {target}")


if __name__ == "__main__":
    main()
