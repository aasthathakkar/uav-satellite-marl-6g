#!/usr/bin/env bash
# Build docs/REPORT.pdf from docs/REPORT.md (needs pandoc + a TeX engine with xelatex).
set -euo pipefail
cd "$(dirname "$0")/../docs"
pandoc REPORT.md -o REPORT.pdf --pdf-engine=xelatex --resource-path=.:.. \
  -V geometry:margin=2cm -V fontsize=10pt -V colorlinks=true \
  -V mainfont="DejaVu Serif" -V sansfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" \
  -V header-includes='\usepackage{float}\floatplacement{figure}{H}'
echo "wrote docs/REPORT.pdf"
