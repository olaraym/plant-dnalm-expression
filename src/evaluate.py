"""Shared prediction head and metrics.

Every feature set (k-mers, each language model's embeddings) goes through the same
ridge-regression head, with the penalty chosen on the validation split. The only thing that
changes between runs is the features, so differences in the test scores come from the features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

DEFAULT_ALPHAS = tuple(10.0 ** np.arange(-2, 5))


def per_output_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Pearson r, Spearman rho and R^2 for each output column, plus their means."""
    y_true = np.atleast_2d(y_true.T).T
    y_pred = np.atleast_2d(y_pred.T).T
    rows = []
    for j in range(y_true.shape[1]):
        t, p = y_true[:, j], y_pred[:, j]
        if np.std(t) == 0 or np.std(p) == 0:
            pr = sr = float("nan")
        else:
            pr = float(pearsonr(t, p)[0])
            sr = float(spearmanr(t, p)[0])
        rows.append({"output": j, "pearson": pr, "spearman": sr, "r2": float(r2_score(t, p))})
    mean = {m: float(np.nanmean([r[m] for r in rows])) for m in ("pearson", "spearman", "r2")}
    return {"per_output": rows, "mean": mean}


def fit_ridge_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> tuple[StandardScaler, Ridge, dict]:
    """Standardize features on train, choose alpha by mean validation Pearson, refit on train."""
    scaler = StandardScaler().fit(X_train)
    Xtr, Xva = scaler.transform(X_train), scaler.transform(X_valid)
    scores = {}
    for a in alphas:
        model = Ridge(alpha=a).fit(Xtr, y_train)
        scores[float(a)] = per_output_metrics(y_valid, model.predict(Xva))["mean"]["pearson"]
    best = max(scores, key=lambda a: -np.inf if np.isnan(scores[a]) else scores[a])
    model = Ridge(alpha=best).fit(Xtr, y_train)
    return scaler, model, {"alpha": best, "valid_pearson_by_alpha": scores}


def evaluate_features(features: dict[str, np.ndarray], task: dict, alphas=DEFAULT_ALPHAS) -> dict:
    """features: {'train': X, 'valid': X, 'test': X}; task: {'train': Split, ...} from pgb.load_task."""
    scaler, model, selection = fit_ridge_head(
        features["train"], task["train"].y, features["valid"], task["valid"].y, alphas
    )
    y_pred = model.predict(scaler.transform(features["test"]))
    return {
        "n_train": int(len(task["train"])),
        "n_valid": int(len(task["valid"])),
        "n_test": int(len(task["test"])),
        "n_features": int(features["train"].shape[1]),
        "n_outputs": int(task["train"].y.shape[1]),
        "selection": selection,
        "test": per_output_metrics(task["test"].y, y_pred),
    }, y_pred


def save_result(result: dict, y_pred: np.ndarray, test_ids: list[str], out_prefix: str | Path) -> None:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2))
    header = "id\t" + "\t".join(f"pred_{j}" for j in range(y_pred.shape[1]))
    lines = [header] + [i + "\t" + "\t".join(f"{v:.5g}" for v in row) for i, row in zip(test_ids, y_pred)]
    out_prefix.with_suffix(".predictions.tsv").write_text("\n".join(lines) + "\n")
