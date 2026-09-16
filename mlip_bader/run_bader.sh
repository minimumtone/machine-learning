#!/usr/bin/env bash
set -euo pipefail

CALC_DIR="${1:-.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CALC_DIR"
chgsum.pl AECCAR0 AECCAR2
bader CHGCAR -ref CHGCAR_sum
python3 "$SCRIPT_DIR/parse_bader.py" "$PWD"
