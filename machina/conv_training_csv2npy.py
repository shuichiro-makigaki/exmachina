from pathlib import Path
import logging

import numpy as np
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


def csv2npy(dim, prefix, n_sample=None):
    csv_list = list(Path(f'data/train/process_part/').glob('*.csv'))
    if n_sample is None:
        n = 0
        logging.info(f'Counting samples...')
        for fcsv in tqdm(csv_list):
            with fcsv.open() as f:
                n += len([x for x in f])
        logging.info(f'{n} samples')
        n_sample = n
    i = 0
    x = np.empty((n_sample, dim), dtype=np.int8)
    y = np.empty(n_sample, dtype=np.int8)
    for fcsv in tqdm(csv_list):
        with fcsv.open() as f:
            for l in f:
                row = l.split(',')
                assert len(row) == dim + 1
                x[i, :] = [int(_) for _ in row[:-1]]
                y[i] = int(float(row[-1]))
                i += 1
    np.save(f'{prefix}_x.npy', x)
    np.save(f'{prefix}_y.npy', y)


def csv2npy_parted(dim, prefix, n_part=None, n_sample=None):
    csv_list = list(Path(f'data/train/scop40_part/').glob('*.csv'))
    if n_sample is None:
        n = 0
        logging.info(f'Counting samples...')
        for fcsv in tqdm(csv_list):
            with fcsv.open() as f:
                n += len([x for x in f])
        logging.info(f'{n} samples')
        n_sample = n
    assert n_sample % n_part == 0
    part_max = int(n_sample / n_part)
    i = 0
    part = 1
    x = np.empty((part_max, dim), dtype=np.int8)
    y = np.empty(part_max, dtype=np.int8)
    for fcsv in tqdm(csv_list):
        with fcsv.open() as f:
            for l in f:
                row = l.split(',')
                try:
                    x[i, :] = [int(_) for _ in row[:-1]]
                    y[i] = int(float(row[-1]))
                except IndexError:
                    np.save(f'{prefix}_x_{part}.npy', x)
                    np.save(f'{prefix}_y_{part}.npy', y)
                    x = np.empty((part_max, dim), dtype=np.int8)
                    y = np.empty(part_max, dtype=np.int8)
                    part += 1
                    i = 0
                    x[i, :] = [int(_) for _ in row[:-1]]
                    y[i] = int(float(row[-1]))
                i += 1
    np.save(f'{prefix}_x_{part}.npy', x)
    np.save(f'{prefix}_y_{part}.npy', y)
