#!/bin/bash -x

if [ ! -e data/train ]; then
  mkdir data/train
fi
cd data/train

BASE=http://www.cb.cs.titech.ac.jp/~makigaki/machina/data
BASE=https://scop.berkeley.edu/downloads/scopseq-1.75

wget $BASE/astral-scopdom-seqres-gd-sel-gs-bib-40-1.75.fa
