#!/usr/bin/env bash
# Prepare the Freddie Mac Single-Family Loan-Level Dataset *sample* files.
#
# The dataset has no anonymous download API: each user registers (free) at
#   https://freddiemac.embs.com/FLoan/
# accepts Freddie Mac's terms, and downloads sample_YYYY.zip files by hand. This script then
# verifies and unpacks them. Raw data never leaves data/ (gitignored) and must not be
# redistributed - see README "Data and licence".
#
# Usage: bash scripts/fetch_data.sh [DOWNLOAD_DIR]      (default: data/downloads)
set -euo pipefail

DOWNLOADS="${1:-data/downloads}"
RAW_DIR="data/raw"
VINTAGES="${VINTAGES:-2010 2011 2012 2013 2014 2015 2016}"
PYTHON="${PYTHON:-python}"

mkdir -p "$RAW_DIR"
missing=()
for y in $VINTAGES; do
  [[ -f "$DOWNLOADS/sample_$y.zip" ]] || missing+=("sample_$y.zip")
done
if (( ${#missing[@]} )); then
  echo "Missing in $DOWNLOADS: ${missing[*]}" >&2
  echo "Register at https://freddiemac.embs.com/FLoan/ , download the sample files for" >&2
  echo "vintages $VINTAGES and place them in $DOWNLOADS (or pass the folder as an argument)." >&2
  exit 1
fi

for y in $VINTAGES; do
  zip="$DOWNLOADS/sample_$y.zip"
  echo "Unpacking $zip"
  "$PYTHON" -m zipfile -t "$zip" >/dev/null   # integrity check
  "$PYTHON" -m zipfile -e "$zip" "$RAW_DIR"
  for f in "sample_orig_$y.txt" "sample_perf_$y.txt"; do
    [[ -s "$RAW_DIR/$f" ]] || { echo "Expected $RAW_DIR/$f after unzip" >&2; exit 1; }
  done
done

# Local manifest so a later run can confirm it used identical inputs (not committed).
( cd "$RAW_DIR" && sha256sum sample_orig_*.txt sample_perf_*.txt > MANIFEST.sha256 )
echo "Done. Files in $RAW_DIR:"
ls -l "$RAW_DIR"
