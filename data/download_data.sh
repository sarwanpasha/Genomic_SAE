#!/bin/bash
# Download all public inputs needed to reproduce the paper.
# Usage: bash data/download_data.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== hg38 reference =="
wget -c -O hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip -f hg38.fa.gz
# index built on first use by pyfaidx

echo "== ENCODE SCREEN candidate cis-regulatory elements =="
wget -c -O encode_ccres_hg38.bed https://api.wenglab.org/screen_v13/fdownloads/GRCh38-cCREs.bed

echo "== JASPAR 2024 CORE vertebrates (PFMs) =="
wget -c -O jaspar2024_core_vertebrates.txt \
  https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt

echo "== ENCODE ChIP-seq IDR peak sets =="
# CTCF GM12878
wget -c -O ctcf_gm12878_idr.bed.gz \
  https://www.encodeproject.org/files/ENCFF796WRU/@@download/ENCFF796WRU.bed.gz
gunzip -f ctcf_gm12878_idr.bed.gz && mv ENCFF796WRU.bed ctcf_gm12878_idr.bed 2>/dev/null || true
# GATA1 K562
wget -c -O gata1_k562_idr.bed.gz \
  https://www.encodeproject.org/files/ENCFF148JKK/@@download/ENCFF148JKK.bed.gz
gunzip -f gata1_k562_idr.bed.gz && mv ENCFF148JKK.bed gata1_k562_idr.bed 2>/dev/null || true
# REST K562
wget -c -O rest_k562_idr.bed.gz \
  https://www.encodeproject.org/files/ENCFF126YZM/@@download/ENCFF126YZM.bed.gz
gunzip -f rest_k562_idr.bed.gz && mv ENCFF126YZM.bed rest_k562_idr.bed 2>/dev/null || true

echo "== done. Model checkpoints download automatically from HuggingFace on first use. =="
