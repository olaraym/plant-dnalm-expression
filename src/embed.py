"""Frozen embeddings from pretrained plant DNA language models.

Each 6 kb input is cut into windows the model can read. Every window's last hidden layer is
mean-pooled over its tokens, and the window vectors are averaged into one vector per sequence.
The model's weights are never updated (frozen); only the ridge head in evaluate.py is trained.

The pooling and aggregation logic is plain numpy, so it is tested without a GPU. The model call
in `embed_sequences` needs torch + transformers (see requirements-gpu.txt).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    window_bp: int
    note: str
    trust_remote_code: bool = True


MODELS = {
    # PlantCaduceus: 16 angiosperm genomes, 512 bp context, single-nucleotide tokens, Apache-2.0.
    # Built on Mamba: needs a CUDA GPU and the mamba-ssm package.
    "plantcaduceus-l20": ModelSpec("kuleshov-group/PlantCaduceus_l20", 512, "20M parameters; fastest"),
    "plantcaduceus-l24": ModelSpec("kuleshov-group/PlantCaduceus_l24", 512, "40M parameters"),
    "plantcaduceus-l32": ModelSpec("kuleshov-group/PlantCaduceus_l32", 512, "225M parameters; largest"),
    # AgroNT: 48 plant species, 1B parameters, 6-mer tokens, 1024-token context (about 6 kb).
    # License CC-BY-NC-SA-4.0 (non-commercial). Needs a GPU with about 16 GB memory.
    "agront-1b": ModelSpec("InstaDeepAI/agro-nucleotide-transformer-1b", 6000, "1B parameters; 6-mer tokens"),
}


# ----------------------------------------------------------------------------- windows
def windows(seq: str, window_bp: int, stride_bp: int | None = None) -> list[str]:
    """Cut a sequence into windows of window_bp. The last window is aligned to the sequence end,
    so every base is covered and all windows have the same length."""
    stride_bp = stride_bp or window_bp
    if len(seq) <= window_bp:
        return [seq]
    starts = list(range(0, len(seq) - window_bp + 1, stride_bp))
    if starts[-1] + window_bp < len(seq):
        starts.append(len(seq) - window_bp)
    return [seq[s : s + window_bp] for s in starts]


# ----------------------------------------------------------------------------- pooling (numpy)
def masked_mean(hidden: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """hidden (batch, tokens, dim), mask (batch, tokens) of 0/1 -> (batch, dim)."""
    if mask is None:
        return hidden.mean(axis=1)
    m = mask[..., None].astype(hidden.dtype)
    return (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)


def aggregate_windows(window_vectors: np.ndarray, owner: np.ndarray, n_sequences: int) -> np.ndarray:
    """Average window vectors belonging to the same sequence. owner[i] = sequence index of window i."""
    dim = window_vectors.shape[1]
    sums = np.zeros((n_sequences, dim), dtype=np.float64)
    np.add.at(sums, owner, window_vectors)
    counts = np.bincount(owner, minlength=n_sequences)[:, None]
    if (counts == 0).any():
        raise ValueError("Some sequences have no windows")
    return (sums / counts).astype(np.float32)


def max_tokens(tokenizer, model) -> int | None:
    """Longest token sequence the model accepts, or None if it reports no limit.

    6-mer tokenizers (AgroNT) split any k-mer containing N into single letters, so a 6 kb window
    with Ns can exceed the model's context; over-long inputs crash on GPU with a device-side assert.
    ESM-style models offset positions by 2, so the usable length is max_position_embeddings - 2.
    """
    limits = []
    mml = getattr(tokenizer, "model_max_length", None)
    if isinstance(mml, int) and 0 < mml < 100_000:   # tokenizers without a limit report a huge number
        limits.append(mml)
    mpe = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(mpe, int) and mpe > 2:
        limits.append(mpe - 2)
    return min(limits) if limits else None


# ----------------------------------------------------------------------------- model call
def load_model(name: str, device: str = "cuda"):
    """Load tokenizer and model for a preset in MODELS (downloads from Hugging Face on first use)."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    spec = MODELS[name]
    tok = AutoTokenizer.from_pretrained(spec.repo, trust_remote_code=spec.trust_remote_code)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForMaskedLM.from_pretrained(
        spec.repo, trust_remote_code=spec.trust_remote_code, torch_dtype=dtype
    ).to(device).eval()
    return tok, model, spec


def embed_sequences(seqs, tokenizer, model, window_bp: int, batch_size: int = 16, device: str = "cuda",
                    progress: bool = True) -> np.ndarray:
    """Frozen mean-pooled embeddings, one row per input sequence."""
    import inspect

    import torch

    all_windows, owner = [], []
    for i, s in enumerate(seqs):
        w = windows(s, window_bp)
        all_windows.extend(w)
        owner.extend([i] * len(w))
    owner = np.asarray(owner)

    accepts_mask = "attention_mask" in inspect.signature(model.forward).parameters
    limit = max_tokens(tokenizer, model)
    n_with_n = sum(any(c not in "ACGT" for c in w) for w in all_windows)
    if progress:
        print(f"{len(all_windows)} windows; {n_with_n} contain non-ACGT bases; token limit = {limit}")
    iterator = range(0, len(all_windows), batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="windows", unit="batch")
        except ImportError:
            pass

    pooled = []
    with torch.inference_mode():
        for start in iterator:
            batch = all_windows[start : start + batch_size]
            if limit:
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=limit)
            else:
                enc = tokenizer(batch, return_tensors="pt", padding=True)
            kwargs = {"input_ids": enc["input_ids"].to(device), "output_hidden_states": True}
            mask = enc.get("attention_mask")
            if accepts_mask and mask is not None:
                kwargs["attention_mask"] = mask.to(device)
            out = model(**kwargs)
            hidden = out.hidden_states[-1].float().cpu().numpy()
            pooled.append(masked_mean(hidden, None if mask is None else mask.numpy()))
    return aggregate_windows(np.vstack(pooled), owner, len(seqs))
