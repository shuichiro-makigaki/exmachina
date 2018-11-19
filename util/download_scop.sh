#!/bin/bash -x

BASE=http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/scop
BASE=http://scop.mrc-lmb.cam.ac.uk/scop/parse

if [ ! -e data/scop ]; then
  mkdir data/scop
fi
cd data/scop

wget $BASE/dir.des.scop.txt_1.75
wget $BASE/dir.cla.scop.txt_1.75
wget $BASE/dir.hie.scop.txt_1.75
wget $BASE/dir.com.scop.txt_1.75

