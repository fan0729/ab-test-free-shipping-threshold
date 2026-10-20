#!/usr/bin/env bash
# Reproduce everything. Usage: bash run_all.sh [--skip-montecarlo]
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data results figures memo
python3 data_gen/generate_experiment.py      # simulated experiment + pilot history + hidden truth
python3 src/01_power.py   > /dev/null        # planning from pilot only
python3 src/02_validity.py > /dev/null       # SRM, balance, A/A
python3 src/03_analysis.py > /dev/null       # main analysis
python3 src/04_peeking.py  > /dev/null       # sequential testing
if [[ "${1:-}" != "--skip-montecarlo" ]]; then
  python3 src/07_monte_carlo_validation.py > results/montecarlo.log   # ~10 min
fi
python3 src/06_reveal_truth.py > /dev/null   # compare estimates with the known truth
python3 src/08_build_docs.py                 # design doc, memo, resume notes
if command -v pandoc >/dev/null && command -v wkhtmltopdf >/dev/null; then
  (cd memo && for f in 01_experiment_design 02_decision_memo; do
     pandoc $f.md -s --embed-resources --resource-path=. --metadata pagetitle="$f" -c style.css -o /tmp/$f.html &&
     wkhtmltopdf -q --enable-local-file-access -s Letter -T 15mm -B 15mm -L 14mm -R 14mm /tmp/$f.html $f.pdf 2>/dev/null; done) \
    || echo "PDF step failed; skipped"
else
  echo "pandoc / wkhtmltopdf not found; skipping PDFs"
fi
(cd notebooks && python3 build_notebook.py > /dev/null 2>&1) || echo "notebook step skipped (needs nbformat, nbconvert, ipykernel)"
echo "done. See memo/ and figures/."
