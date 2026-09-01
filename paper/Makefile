# Paper build. Requires: tectonic, and .venv (python: matplotlib, numpy, pyyaml).
PY := ./.venv/bin/python
MAIN := main

.PHONY: all assets pdf check arxiv clean

all: assets pdf check

assets:
	$(PY) scripts/gen_wandb.py
	$(PY) scripts/gen_tables.py
	$(PY) scripts/gen_figures.py
	$(PY) ../eval/aggregate.py
	$(PY) ../eval/calc_hardware.py

pdf: assets
	tectonic -X compile $(MAIN).tex --keep-intermediates

check: pdf
	$(PY) scripts/check_numbers.py

# arXiv-ready tarball: flatten sources + precompiled .bbl, then clean-room compile.
arxiv: pdf
	rm -rf arxiv_build && mkdir -p arxiv_build/sections arxiv_build/tables arxiv_build/figures
	cp $(MAIN).tex references.bib arxiv_build/
	cp sections/*.tex arxiv_build/sections/
	cp tables/*.tex arxiv_build/tables/
	cp figures/*.pdf arxiv_build/figures/
	cp $(MAIN).bbl arxiv_build/ 2>/dev/null || true
	cd arxiv_build && tar czf ../$(MAIN)_arxiv.tar.gz .
	@echo "Wrote $(MAIN)_arxiv.tar.gz"

clean:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg $(MAIN).out
	rm -rf arxiv_build $(MAIN)_arxiv.tar.gz
