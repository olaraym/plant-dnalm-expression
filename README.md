# Do plant DNA language models beat k-mers at predicting gene expression?

[![CI](https://github.com/Olaraym/plant-dnalm-expression/actions/workflows/ci.yml/badge.svg)](https://github.com/Olaraym/plant-dnalm-expression/actions/workflows/ci.yml)

**Question.** How well do *frozen* embeddings from pretrained plant DNA language models predict tissue-level
gene expression from a gene's 6 kb regulatory sequence, compared with a simple k-mer frequency baseline?

**Why it matters.** Language models are increasingly used to score regulatory variants and design sequences
for crops. A model is only useful for that if it captures more than a k-mer count already does.

**Status:** v0.1: first comparison complete on *Arabidopsis thaliana* (Nov 2026). Next steps are in [PLAN.md](PLAN.md).

**Headline:** frozen embeddings from the plant DNA language model AgroNT predicted expression across 56 tissues and
conditions far better than k-mer features: mean test Spearman ρ **0.577 vs. 0.339** on the same 5,000 training genes.
They also beat k-mers trained on five times more genes (0.377). The gain is 0.238 (95% bootstrap CI 0.212–0.264),
and AgroNT wins on all 56 outputs.

## Data

| Dataset | Source | Citation | License |
| --- | --- | --- | --- |
| Plants Genomic Benchmark, gene expression task (6 kb sequences, multi-output regression; fixed train/valid/test splits per species) | [InstaDeepAI/plant-genomic-benchmark](https://huggingface.co/datasets/InstaDeepAI/plant-genomic-benchmark) | Mendoza-Revilla et al. (2024) *Communications Biology* | CC-BY-NC-SA-4.0 |

Species: *Arabidopsis thaliana*, the smallest gene-expression task. Confirmed on download:

- **Splits:** 25,731 train / 3,401 validation / 3,402 test genes (fixed by the benchmark)
- **Input:** 6,000 bp per gene; every sequence is exactly 6,000 bp
- **Output:** 56 expression values per gene (tissues / conditions), stored in the FASTA header as
  `GENE_ID|value_1|...|value_56`. Values are used as provided by PGB (they appear to be on a log scale).

Raw data are not committed; `make data` downloads them and prints the label layout.

## Models compared

All feature sets go through **the same ridge-regression head**. Features are standardized, and the ridge penalty
(alpha, searched from 0.01 to 10⁷) is chosen by mean validation Pearson r. Only the features differ between runs.
Every model uses the same 5,000 training genes, drawn at random with a fixed seed, and the full validation and test splits.

| Features | What it is | Dimensions |
| --- | --- | --- |
| k-mer (k = 4, 5, 6) | normalized k-mer frequencies | 256 / 1,024 / 4,096 |
| AgroNT, frozen | 1B-parameter transformer pre-trained on 48 plant genomes ([Mendoza-Revilla et al. 2024](https://doi.org/10.1038/s42003-024-06465-2)); last hidden layer mean-pooled over one 6 kb window (6-mer tokens) | 1,500 |
| PlantCaduceus, frozen | Mamba-based DNA LM, 16 angiosperm genomes ([Zhai et al. 2024](https://www.biorxiv.org/content/10.1101/2024.06.04.596709)) | not run yet (see Limitations) |

Metrics on the held-out test split, per expression output and averaged over the 56 outputs: Pearson r, Spearman ρ, R².

## Results

| Features | Training genes | Mean Spearman ρ (test) | Mean Pearson r (test) | Mean R² (test) | Ridge alpha |
| --- | --- | --- | --- | --- | --- |
| **AgroNT, frozen** | 5,000 | **0.577** | **0.568** | **0.313** | 100 |
| k-mer k=6 | 5,000 | 0.339 | 0.346 | 0.088 | 100,000 |
| k-mer k=5 | 5,000 | 0.330 | 0.336 | 0.111 | 10,000 |
| k-mer k=4 | 5,000 | 0.317 | 0.318 | 0.100 | 1,000 |
| k-mer k=6 (reference: all training genes) | 25,731 | 0.377 | 0.381 | 0.137 | 100,000 |

![Per-output test Spearman, one dot per expression output](results/figures/main.png)

Source files: [`results/summary.csv`](results/summary.csv) (one row per model) and
[`results/summary_per_output.csv`](results/summary_per_output.csv) (one row per model and output).

**What the results show**

- **A large gain from pretraining.** On identical training data, AgroNT raised mean Spearman from 0.339 to 0.577,
  and R² from 0.088 to 0.313: about 3.5 times more of the between-gene variation in expression explained.
- **Not a fluke of the test set.** Resampling the 3,402 test genes 300 times gives a gain of **0.238, 95% CI 0.212–0.264**.
  The interval is far from zero.
- **Consistent across tissues.** AgroNT wins on **56 of 56** outputs, with per-output gains from 0.12 to 0.28 (median 0.24).
  Its lowest output (ρ = 0.449) beats the highest output of every k-mer model, including k = 6 trained on all
  25,731 genes (0.429).
- **Pretraining beats more data.** AgroNT with 5,000 genes beats k-mers trained on 25,731 genes (0.577 vs. 0.377).
- **k-mer size trades off against data.** With 5,000 genes, k=6 ranks genes best but has a lower R² than k=5: heavy
  regularization (alpha = 100,000) shrinks its predictions toward the mean.

**Checks**

- The k=6 baseline gave identical scores on powershell run on my local computer and on Colab, so the pipeline is reproducible across machines.
- [`notebooks/local_walkthrough.ipynb`](notebooks/local_walkthrough.ipynb) reruns every step on a laptop from the saved
  embeddings and reproduces the table exactly; it also computes the per-output comparison and the bootstrap interval
  ([`results/walkthrough/`](results/walkthrough/)).
- The first k-mer runs chose the largest alpha in the original grid (10⁴). The grid was widened to 10⁷ and the code
  now warns if the chosen alpha sits on an edge; every alpha in the table is inside the grid.

**Limitations**

- One species, one random training subset, one seed.
- Frozen features with a linear head: no fine-tuning. The published AgroNT results come from fine-tuning, so they are
  an upper reference, not a like-for-like comparison.
- 7 of the 6,803 validation and test windows contain non-ACGT bases. With 6-mer tokens these exceed AgroNT's
  1,024-token context and were truncated at that limit.
- PlantCaduceus was not run: its `mamba-ssm` and `causal-conv1d` CUDA packages had no prebuilt build for the free Colab
  runtime and did not finish compiling after 35 minutes. Planned on the NC State HPC cluster.

**What the predictions show** ([`results/walkthrough/walkthrough_comparison.png`](results/walkthrough/walkthrough_comparison.png))

- Many test genes have observed expression of exactly 0, and the lowest non-zero values fall on a few discrete levels
  (about 0.69, 1.10, 1.39), a pattern consistent with a log(count + 1) transform. Both models struggle to pick out the
  unexpressed genes.
- AgroNT's predictions span a wider range than the k-mer model's (whose strong penalty shrinks them toward the mean),
  which is why its R² is much higher. As a linear model it can predict values below 0, which are impossible on this scale.

**Compute**

AgroNT embeddings ran on a free Google Colab T4 GPU (half precision, batch size 8, about 3.8 s per batch): 39 minutes
for 5,000 training genes and about 27 minutes for each of the validation and test splits. The k-mer features and all
ridge fits run on a laptop CPU in minutes.

## Reproduce

```bash
git clone https://github.com/Olaraym/plant-dnalm-expression.git
cd plant-dnalm-expression
make env && conda activate plant-dnalm-expression
make test                      # unit tests + offline end-to-end run on synthetic data
make data                      # download the Arabidopsis task and print its layout
make baseline MAX_TRAIN=5000   # k-mer baselines (CPU, a few minutes)
```

Embedding extraction needs a GPU. Open
[`notebooks/colab_extract_embeddings.ipynb`](notebooks/colab_extract_embeddings.ipynb) in Google Colab
(T4 GPU) and use `MODEL = "agront-1b"`. Save embeddings to Google Drive so a session timeout cannot lose
finished splits:

```bash
python scripts/extract_embeddings.py --species arabidopsis_thaliana --model agront-1b --max-train 5000 \
    --batch-size 8 --emb-dir /content/drive/MyDrive/pgb_embeddings
python scripts/run_embedding_ridge.py --species arabidopsis_thaliana --model agront-1b --max-train 5000 \
    --emb-dir /content/drive/MyDrive/pgb_embeddings
python scripts/compare.py
```

On Windows without `make`, run the commands inside the Makefile directly with your environment's Python.

To see every step's input and output on a laptop (no GPU needed once the embeddings are saved), open
[`notebooks/local_walkthrough.ipynb`](notebooks/local_walkthrough.ipynb).

## Layout

```
src/pgb.py        download + parse PGB FASTA files (labels in '|'-separated headers)
src/kmers.py      k-mer frequency features
src/embed.py      windowing, pooling and frozen-model embedding extraction
src/evaluate.py   shared ridge head (alpha chosen on validation) and metrics
scripts/          command-line steps (download, baseline, embeddings, ridge, compare, synthetic data)
notebooks/        Colab notebook for the GPU step
tests/            pytest suite, including an end-to-end run on synthetic motif-driven data
```

## Author

Laide A. Rasaki · NC State University · [ORCID 0000-0002-5594-6827](https://orcid.org/0000-0002-5594-6827)
AI assistance is documented in [AI_ASSISTANCE.md](AI_ASSISTANCE.md).
