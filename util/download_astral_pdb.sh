#!/bin/bash -x

if [[ ! -e data/train ]]; then
  mkdir data/train
fi
cd data/train

wget http://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-1.75.tgz
tar zxf pdbstyle-sel-gs-bib-40-1.75.tgz && rm pdbstyle-sel-gs-bib-40-1.75.tgz
