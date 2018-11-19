from pathlib import Path
import pickle
import logging
import random
import datetime

from sklearn.model_selection import KFold
import numpy as np
from Bio.SCOP import Scop
from Bio import SeqIO

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

n_splits = 1
sf_sunid = 52540
test_n_splits = 4

test_data = (
    ('d1wlqc_', datetime.datetime(2009, 2, 17, 0, 0), 'a.4.5', 762),
    ('d2axtu1', datetime.datetime(2009, 2, 10, 0, 0), 'a.60.12', 159),
    ('d2zqna1', datetime.datetime(2009, 2, 10, 0, 0), 'b.42.2', 119),
    ('d1qg3a1', datetime.datetime(2009, 1, 20, 0, 0), 'b.1.2', 344),
    ('d1wzca1', datetime.datetime(2009, 1, 27, 0, 0), 'c.108.1', 296),
    ('d2dsta1', datetime.datetime(2009, 1, 27, 0, 0), 'c.69.1', 975),
    ('d1y5ha3', datetime.datetime(2009, 2, 10, 0, 0), 'd.37.1', 62),
    ('d2pzza1', datetime.datetime(2009, 1, 20, 0, 0), 'd.77.1', 92),
    ('d1ni9a_', datetime.datetime(2009, 2, 10, 0, 0), 'e.7.1', 151),
    ('d3cw9a1', datetime.datetime(2008, 9, 2, 0, 0), 'e.23.1', 22),
    ('d2axtd1', datetime.datetime(2009, 2, 10, 0, 0), 'f.26.1', 174),
    ('d2axto1', datetime.datetime(2009, 2, 10, 0, 0), 'f.4.1', 15),
    ('d2vy4a1', datetime.datetime(2009, 2, 17, 0, 0), 'g.37.1', 182),
    ('d3d9ta1', datetime.datetime(2009, 2, 10, 0, 0), 'g.52.1', 81),
)
test_data = [x[0] for x in test_data]

scop40 = SeqIO.index('data/astral-scopdom-seqres-gd-sel-gs-bib-40-1.75.fa', 'fasta')
scop100_hie = Scop(dir_path=Path('data/scop'), version='1.75')

if n_splits > 1:
    fold1 = next(KFold(n_splits=n_splits, shuffle=True).split(scop40))
    samples = np.array([v for i, v in enumerate(scop40) if i in fold1[1]])
    # sf_sunid of scop100 (sid)
    px = np.array([x.sid for x in scop100_hie.getNodeBySunid(sf_sunid).getDescendents('px')])
    # select only sf_sunid in scop40 from scop100
    isect = np.intersect1d(samples, px)
    # select half of sf_sunid only in scop40 for test data
    fold1 = next(KFold(n_splits=test_n_splits, shuffle=True).split(isect))
    tests = np.array([v for i, v in enumerate(isect) if i in fold1[1]])
    np.save(Path(f'data/test/scop40_{n_splits}fold_sf{sf_sunid}_testdata_{test_n_splits}fold.npy'), tests)
    # select domain sids only in scop40 for training data
    train = np.setdiff1d(samples, tests)
    np.save(Path(f'data/train/scop40_{n_splits}fold_trainingdata.npy'), train)
    # {sf: [sid]} list for making alignment pairs in the same superfamily
    hie = {}
    for i in train:
        dom = scop100_hie.getDomainBySid(i)
        if dom:
            sf = dom.getAscendent('sf').sccs
        else:
            # FIX: Why nothing?
            continue
        if sf in hie:
            hie[sf].append(i)
        else:
            hie[sf] = [i]
    pickle.dump(hie, Path(f'data/train/scop40_{n_splits}fold_sf{sf_sunid}_hie.pkl').open('wb'))
else:
    train = np.array([x for x in scop40 if x not in test_data])
    np.save(Path(f'data/train/scop40_{n_splits}fold.npy'), train)
    test = np.array(test_data)
    np.save(Path(f'data/test/scop40_{n_splits}fold.npy'), test)
    hie = {}
    for sf in scop100_hie.getRoot().getDescendents('sf'):
        isect = np.intersect1d(train, np.array([x.sid for x in sf.getDescendents('px')]))
        hie[sf.sunid] = isect.tolist()
    pickle.dump(hie, Path(f'data/train/scop40_{n_splits}fold_hie.pkl').open('wb'))
