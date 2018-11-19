#!/bin/sh
#$ -l q_core=1
#$ -l h_rt=12:00:00
#$ -j y

. /etc/profile.d/modules.sh
module load python/3.6.5

python3 evaluate_model.py $1
