# Two-week sprint: frozen language-model features vs. k-mers (Nov 23 – Dec 6, 2026)

Goal: a public, tested repo with one honest comparison on real data, live before internship interviews.
Budget: about 8 hours a week. Tick boxes as you go and mirror them as issues on the project board.

## Week 1: data and baseline (Nov 23 – 29)

- [ ] **Day 1 (1 h).** Create the repo on GitHub (setup script or `gh repo create`), push this code, and confirm the CI badge is green.
- [ ] **Day 2 (1 h).** `make env`, then `make data`. Check the printed headers: number of outputs (tissues), label scale (raw or log), and sequence lengths. Write these facts into the README's Data section.
- [ ] **Day 3 (2 h).** `make baseline MAX_TRAIN=5000`, then the same with the full training set (`MAX_TRAIN=` with no value means all). Record the k=4, 5, 6 test scores in `CHANGELOG.md`.
- [ ] **Days 4–5 (3 h).** In Colab (T4 GPU), run `notebooks/colab_extract_embeddings.ipynb` with PlantCaduceus-l20 and `MAX_TRAIN = 5000`. If `mamba-ssm` fails to install, use AgroNT with `MAX_TRAIN = 2000` and rerun the baseline with the same value.

## Week 2: comparison and release (Nov 30 – Dec 6)

- [ ] **Day 6 (1 h).** Unzip the Colab outputs into the repo and run `make compare`. Check that `results/summary.csv` has both feature sets on identical `n_train` and `n_test`.
- [ ] **Day 7 (2 h).** One extension, only if on time: a larger PlantCaduceus (l32), or a second species (rice).
- [ ] **Day 8 (2 h).** Write the Results section: table, figure, three sentences of interpretation. Look up the published AgroNT gene-expression results and quote them as the fine-tuned reference, citing the paper and table number.
- [ ] **Day 9 (1 h).** Fill in `AI_ASSISTANCE.md` (what you checked by hand), tag `v0.1.0`, archive on Zenodo, and update the resume bullet below with the real numbers.

## Resume bullet (fill in after Day 8)

> Benchmarked frozen plant DNA language-model embeddings (PlantCaduceus) against k-mer features for predicting
> tissue-specific gene expression from 6 kb regulatory sequences in *Arabidopsis* ([N] genes); best model reached
> mean test Spearman ρ = [X] vs. [Y] for the k-mer baseline (PyTorch, Hugging Face, scikit-learn; tested, CI, Zenodo DOI).

## Done means

- [ ] CI green
- [ ] Results table and figure in README, from files in `results/`
- [ ] Same training subset, splits and ridge head for every feature set
- [ ] Interpretation states the limits: frozen embeddings, ridge head, one species
- [ ] `AI_ASSISTANCE.md` filled in
- [ ] Release v0.1.0 with a DOI
