"""Download and parse the gene-expression tasks of the Plants Genomic Benchmark (PGB).

PGB (InstaDeepAI/plant-genomic-benchmark on Hugging Face) stores each task as FASTA files.
Each header holds a sequence id followed by '|'-separated labels, e.g.

    >AT1G01010_0|2.31|1.87|0.55|...
    ACGT...

For gene_exp tasks the labels are expression values for several tissues (multi-output regression).
`parse_header` also accepts labels joined by commas or whitespace inside one field, so it keeps
working if the exact layout differs. Run `scripts/download_data.py` first: it prints the first
headers so you can confirm the layout before any modelling.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ID = "InstaDeepAI/plant-genomic-benchmark"
SPECIES = (
    "arabidopsis_thaliana",
    "glycine_max",
    "oryza_sativa",
    "solanum_lycopersicum",
    "zea_mays",
)
SPLITS = ("train", "valid", "test")
_SPLIT_ALIASES = {
    "train": ("train",),
    "valid": ("valid", "val", "validation", "dev"),
    "test": ("test",),
}
_FASTA_EXT = (".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz")


@dataclass
class Split:
    ids: list[str]
    seqs: list[str]
    y: np.ndarray  # shape (n_sequences, n_outputs)

    def __len__(self) -> int:
        return len(self.seqs)

    def subset(self, idx: np.ndarray) -> "Split":
        return Split([self.ids[i] for i in idx], [self.seqs[i] for i in idx], self.y[idx])


# ----------------------------------------------------------------------------- parsing
def parse_header(header: str) -> tuple[str, list[float]]:
    """Split a FASTA header (without '>') into (id, labels)."""
    fields = header.strip().split("|")
    seq_id = fields[0].strip()
    tokens: list[str] = []
    for field in fields[1:]:
        tokens.extend(t for t in re.split(r"[,\s;]+", field.strip()) if t)
    if not tokens:
        raise ValueError(f"No labels found in header: {header[:80]!r}")
    return seq_id, [float(t) for t in tokens]


def _open(path: Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "rt")


def read_fasta(path: str | Path) -> Split:
    """Read a PGB FASTA file (single- or multi-line sequences, optionally gzipped)."""
    ids, seqs, labels = [], [], []
    current: list[str] | None = None
    with _open(Path(path)) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    seqs.append("".join(current).upper())
                seq_id, lab = parse_header(line[1:])
                ids.append(seq_id)
                labels.append(lab)
                current = []
            else:
                if current is None:
                    raise ValueError(f"{path}: sequence before the first header")
                current.append(line)
    if current is not None:
        seqs.append("".join(current).upper())

    n_out = {len(l) for l in labels}
    if len(n_out) != 1:
        raise ValueError(f"{path}: inconsistent number of labels per record: {sorted(n_out)}")
    return Split(ids, seqs, np.asarray(labels, dtype=np.float32))


# ----------------------------------------------------------------------------- files
def match_split_files(species: str, filenames: list[str]) -> dict[str, str]:
    """Pick the train/valid/test FASTA files of one species' gene_exp task from a file list."""
    out: dict[str, str] = {}
    candidates = [
        f for f in filenames
        if "gene_exp" in f and species in Path(f).name and f.endswith(_FASTA_EXT)
    ]
    for split, aliases in _SPLIT_ALIASES.items():
        hits = [
            f for f in candidates
            if any(re.search(rf"(^|[_\-.]){a}([_\-.]|$)", Path(f).name) for a in aliases)
        ]
        if len(hits) > 1:
            raise ValueError(f"Several files match {species}/{split}: {hits}")
        if hits:
            out[split] = hits[0]
    missing = [s for s in SPLITS if s not in out]
    if missing:
        raise FileNotFoundError(
            f"No gene_exp file for {species} split(s) {missing}. Candidates: {candidates}"
        )
    return out


def download(species: str, out_dir: str | Path = "data/raw") -> dict[str, Path]:
    """Download one species' gene_exp FASTA files from Hugging Face into out_dir."""
    from huggingface_hub import hf_hub_download, list_repo_files  # imported lazily

    if species not in SPECIES:
        raise ValueError(f"Unknown species {species!r}; choose from {SPECIES}")
    files = match_split_files(species, list_repo_files(REPO_ID, repo_type="dataset"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        split: Path(hf_hub_download(REPO_ID, fname, repo_type="dataset", local_dir=out_dir))
        for split, fname in files.items()
    }


def local_files(species: str, data_dir: str | Path = "data/raw") -> dict[str, Path]:
    """Find already-downloaded files for a species under data_dir."""
    names = [str(p.relative_to(data_dir)) for p in Path(data_dir).rglob("*") if p.is_file()]
    return {s: Path(data_dir) / f for s, f in match_split_files(species, names).items()}


# ----------------------------------------------------------------------------- loading
def subsample_indices(n: int, max_n: int | None, seed: int) -> np.ndarray:
    """Deterministic subset, so every model sees exactly the same sequences."""
    if max_n is None or max_n >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def load_task(
    species: str,
    data_dir: str | Path = "data/raw",
    max_train: int | None = None,
    seed: int = 2026,
) -> dict[str, Split]:
    """Load train/valid/test for a species. Only the training split is subsampled."""
    files = local_files(species, data_dir)
    task = {s: read_fasta(p) for s, p in files.items()}
    task["train"] = task["train"].subset(subsample_indices(len(task["train"]), max_train, seed))
    return task
