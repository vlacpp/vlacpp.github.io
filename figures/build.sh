#!/bin/sh
# Rebuilds ../assets/teaser.svg from teaser.tex. Needs tectonic, poppler's pdftocairo and python3.
set -e
cd "$(dirname "$0")"
tmp=$(mktemp -d)
tectonic --outdir "$tmp" teaser.tex
pdftocairo -svg "$tmp/teaser.pdf" "$tmp/teaser.svg"
# three decimals is far below a pixel and trims about a sixth of the file
python3 -c '
import re, sys
s = open(sys.argv[1]).read()
def r(m):
    v = f"{float(m.group(0)):.3f}".rstrip("0").rstrip(".")
    return "0" if v in ("-0", "") else v
open(sys.argv[2], "w").write(re.sub(r"-?\d+\.\d{4,}", r, s))
' "$tmp/teaser.svg" ../assets/teaser.svg
rm -rf "$tmp"
