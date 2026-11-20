import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import embed  # noqa: E402
import evaluate  # noqa: E402
import kmers  # noqa: E402
import pgb  # noqa: E402

TOY = ROOT / "tests" / "data" / "toy.fa"


# ----------------------------------------------------------------------------- parsing
def test_parse_header_pipe_separated():
    assert pgb.parse_header("AT1G01010_0|1.5|-0.25|3") == ("AT1G01010_0", [1.5, -0.25, 3.0])


def test_parse_header_comma_or_space_inside_one_field():
    assert pgb.parse_header("g1|1.0,2.0 3.0")[1] == [1.0, 2.0, 3.0]


def test_parse_header_without_labels_fails():
    with pytest.raises(ValueError):
        pgb.parse_header("g1")


def test_read_fasta_multiline_and_case(tmp_path):
    s = pgb.read_fasta(TOY)
    assert s.ids == ["geneA_0", "geneB_1", "geneC_2"]
    assert s.seqs[0] == "ACGTACGTAA"          # two lines joined
    assert s.seqs[2] == "NNACGT"              # lower case upper-cased
    assert s.y.shape == (3, 2)


def test_read_fasta_gzip(tmp_path):
    gz = tmp_path / "toy.fa.gz"
    gz.write_bytes(gzip.compress(TOY.read_bytes()))
    assert len(pgb.read_fasta(gz)) == 3


def test_inconsistent_label_counts_fail(tmp_path):
    bad = tmp_path / "bad.fa"
    bad.write_text(">a|1|2\nACGT\n>b|1\nACGT\n")
    with pytest.raises(ValueError):
        pgb.read_fasta(bad)


def test_match_split_files_picks_species_and_splits():
    names = [
        "gene_exp/zea_mays_train.fa", "gene_exp/zea_mays_valid.fa", "gene_exp/zea_mays_test.fa",
        "gene_exp/oryza_sativa_train.fa", "poly_a/zea_mays_train.fa", "README.md",
    ]
    got = pgb.match_split_files("zea_mays", names)
    assert got == {"train": "gene_exp/zea_mays_train.fa", "valid": "gene_exp/zea_mays_valid.fa",
                   "test": "gene_exp/zea_mays_test.fa"}


def test_subsample_is_deterministic():
    a = pgb.subsample_indices(100, 10, seed=1)
    assert np.array_equal(a, pgb.subsample_indices(100, 10, seed=1))
    assert len(a) == 10 and len(set(a)) == 10
    assert np.array_equal(pgb.subsample_indices(5, None, 1), np.arange(5))


# ----------------------------------------------------------------------------- k-mers
def test_kmer_frequencies_known_counts():
    f = kmers.kmer_frequencies("ACGTA", 2)
    names = kmers.kmer_names(2)
    got = {names[i]: v for i, v in enumerate(f) if v > 0}
    assert got == pytest.approx({"AC": 0.25, "CG": 0.25, "GT": 0.25, "TA": 0.25})


def test_kmer_windows_with_n_are_skipped():
    f = kmers.kmer_frequencies("ACNGT", 2)
    names = kmers.kmer_names(2)
    assert {names[i] for i, v in enumerate(f) if v > 0} == {"AC", "GT"}


def test_kmer_matrix_shape():
    X = kmers.kmer_matrix(["ACGTACGT", "TTTT"], ks=(2, 3))
    assert X.shape == (2, 16 + 64)


# ----------------------------------------------------------------------------- metrics and head
@pytest.mark.filterwarnings("ignore:Best ridge alpha")
def test_ridge_head_recovers_linear_signal():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(20, 3))
    X = {s: rng.normal(size=(n, 20)) for s, n in (("train", 400), ("valid", 100), ("test", 100))}

    class S:  # minimal stand-in for pgb.Split
        def __init__(self, y):
            self.y = y

        def __len__(self):
            return len(self.y)

    task = {s: S(X[s] @ W + rng.normal(0, 0.1, size=(len(X[s]), 3))) for s in X}
    result, pred = evaluate.evaluate_features(X, task)
    assert pred.shape == (100, 3)
    assert result["test"]["mean"]["pearson"] > 0.95


def test_metrics_handle_constant_column():
    m = evaluate.per_output_metrics(np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]),
                                    np.array([[1.1, 0.5], [2.1, 0.7], [2.9, 0.9]]))
    assert m["per_output"][0]["pearson"] == pytest.approx(1.0, abs=0.01)
    assert np.isnan(m["per_output"][1]["pearson"])


# ----------------------------------------------------------------------------- embedding helpers
def test_windows_cover_sequence_with_equal_lengths():
    seq = "A" * 1300
    w = embed.windows(seq, 512)
    assert [len(x) for x in w] == [512, 512, 512]   # last window aligned to the end
    assert embed.windows("ACGT", 512) == ["ACGT"]


def test_masked_mean_ignores_padding():
    hidden = np.array([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = np.array([[1, 1, 0]])
    assert embed.masked_mean(hidden, mask).tolist() == [[2.0, 2.0]]


def test_aggregate_windows_averages_per_sequence():
    vecs = np.array([[1.0], [3.0], [10.0]])
    out = embed.aggregate_windows(vecs, np.array([0, 0, 1]), 2)
    assert out.ravel().tolist() == [2.0, 10.0]


def test_model_presets_have_windows():
    assert all(spec.window_bp > 0 for spec in embed.MODELS.values())


# ----------------------------------------------------------------------------- end to end
def test_end_to_end_kmer_baseline_on_synthetic(tmp_path):
    data, results = tmp_path / "data", tmp_path / "results"
    run = lambda *a: subprocess.run([sys.executable, *a], check=True, cwd=ROOT, capture_output=True, text=True)
    run("scripts/make_synthetic.py", "--out", str(data), "--n", "300", "80", "80", "--length", "800")
    out = run("scripts/run_kmer_baseline.py", "--data-dir", str(data), "--results-dir", str(results), "--k", "6")
    assert "test mean Pearson" in out.stdout
    import json
    r = json.loads(next(results.glob("kmer_k6__*.json")).read_text())
    assert r["test"]["mean"]["pearson"] > 0.5   # motif signal must be recovered
    run("scripts/compare.py", "--results-dir", str(results))
    assert (results / "summary.csv").exists() and (results / "figures" / "main.png").exists()


def test_alpha_at_grid_edge_is_flagged():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 5))
    y = rng.normal(size=(60, 2))          # pure noise: the strongest penalty wins
    with pytest.warns(UserWarning, match="edge"):
        _, _, sel = evaluate.fit_ridge_head(X[:40], y[:40], X[40:], y[40:], alphas=(0.1, 1.0, 10.0))
    assert sel["alpha_at_grid_edge"] is True
