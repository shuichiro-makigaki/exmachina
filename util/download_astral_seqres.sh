#!/bin/bash -x

if [[ ! -e data/train ]]; then
  mkdir data/train
fi
cd data/train

wget https://scop.berkeley.edu/downloads/scopseq-1.75/astral-scopdom-seqres-gd-sel-gs-bib-40-1.75.fa
