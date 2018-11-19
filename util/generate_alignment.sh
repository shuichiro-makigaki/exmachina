#!/bin/sh

python3 generate_alignment.py 25 9 $1 $2 &
python3 generate_alignment.py 25 18 $1 $2 &
python3 generate_alignment.py 25 25 $1 $2 &

wait
