#!/bin/bash -x

BASE=http://scop.mrc-lmb.cam.ac.uk/scop/parse

if [[ ! -e data/train ]]; then
  mkdir data/train
fi
cd data/train

wget ${BASE}/dir.des.scop.txt_1.75
wget ${BASE}/dir.cla.scop.txt_1.75
wget ${BASE}/dir.hie.scop.txt_1.75
wget ${BASE}/dir.com.scop.txt_1.75

