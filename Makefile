.PHONY: env data baseline compare smoke test clean

SPECIES   ?= arabidopsis_thaliana
MAX_TRAIN ?= 5000
TRAIN_ARG = $(if $(MAX_TRAIN),--max-train $(MAX_TRAIN),)

env:        ## create the conda environment
	conda env create -f environment.yml || conda env update -f environment.yml

data:       ## download the PGB gene_exp task for SPECIES and print its layout
	python scripts/download_data.py --species $(SPECIES)

baseline:   ## k-mer + ridge baselines (k = 4, 5, 6) on the same training subset as the GPU step
	for k in 4 5 6; do python scripts/run_kmer_baseline.py --species $(SPECIES) $(TRAIN_ARG) --k $$k; done

compare:    ## summary table + figure from every results/*.json
	python scripts/compare.py

smoke:      ## offline end-to-end check on synthetic data (no download, no GPU)
	python scripts/make_synthetic.py --out data/synthetic
	python scripts/run_kmer_baseline.py --data-dir data/synthetic --results-dir results/synthetic
	python scripts/compare.py --results-dir results/synthetic

test:       ## unit + end-to-end tests (used by CI)
	pytest -q tests

clean:
	rm -rf data/synthetic results/synthetic .pytest_cache
