#!/usr/bin/env bash
# Build the technical report: pandoc (markdown -> LaTeX) + pdflatex (LaTeX -> PDF).
# Needs pandoc (in the ml_lab conda env) and a TeX Live with pdflatex (system /usr/bin/pdflatex is enough).
# Run from anywhere:  bash report/build_pdf.sh
set -euo pipefail
cd "$(dirname "$0")"
PANDOC=${PANDOC:-$(command -v pandoc || echo "$HOME/miniconda3/envs/ml_lab/bin/pandoc")}
PDFLATEX=${PDFLATEX:-/usr/bin/pdflatex}
mkdir -p build
"$PANDOC" report.md --from markdown --to latex --standalone --template template.tex --number-sections -o build/report.tex
for i in 1 2; do "$PDFLATEX" -interaction=nonstopmode -halt-on-error -output-directory build build/report.tex > build/pdflatex.log 2>&1 || { tail -30 build/pdflatex.log; exit 1; }; done
cp build/report.pdf PathSnap_Technical_Report.pdf
echo "written report/PathSnap_Technical_Report.pdf ($(grep -c '^' build/report.toc) toc lines)"
