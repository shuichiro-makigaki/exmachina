import logging

import numpy as np
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


def random_sampling(prefix, ratio):
    result_x = []
    result_y = []
    x = np.load(f'{prefix}_x.npy')
    y = np.load(f'{prefix}_y.npy')
    r = np.random.randint(0, high=len(x), size=int(len(x)*ratio))
    result_x.extend(x[r].tolist())
    result_y.extend(y[r].tolist())
    np.save(f'{prefix}_randomsampling_ratio{ratio}_x.npy', np.array(result_x, dtype=np.int8))
    np.save(f'{prefix}_randomsampling_ratio{ratio}_y.npy', np.array(result_y, dtype=np.int8))


def random_sampling_parted(fold, prefix='data/train/scop40_1fold_w9', parts=10):
    result_x = []
    result_y = []
    for i in tqdm(range(1, parts+1)):
        x = np.load(f'{prefix}_x_{i}.npy')
        y = np.load(f'{prefix}_y_{i}.npy')
        r = np.random.randint(0, high=len(x), size=int(len(x)/fold))
        result_x.extend(x[r].tolist())
        result_y.extend(y[r].tolist())
    np.save(f'{prefix}_randomsampling_{fold}fold_x.npy', np.array(result_x, dtype=np.int8))
    np.save(f'{prefix}_randomsampling_{fold}fold_y.npy', np.array(result_y, dtype=np.int8))


def negative_down_sampling(prefix, parts):
    for i in tqdm(range(1, parts+1)):
        result_x = []
        result_y = []
        x = np.load(f'{prefix}_x_{i}.npy')
        y = np.load(f'{prefix}_y_{i}.npy')
        positive_arg = np.argwhere(y == 1)
        result_x.extend(x[positive_arg].tolist())
        result_y.extend(y[positive_arg].tolist())
        negative_arg = np.argwhere(y == 0)
        np.random.shuffle(negative_arg)
        result_x.extend(x[negative_arg[:len(positive_arg)]].tolist())
        result_y.extend(y[negative_arg[:len(positive_arg)]].tolist())
        np.save(f'{prefix}_downsampling_x_{i}.npy', np.array(result_x, dtype=np.int8))
        np.save(f'{prefix}_downsampling_y_{i}.npy', np.array(result_y, dtype=np.int8))
    out_x = np.load(f'{prefix}_downsampling_x_1.npy')
    out_y = np.load(f'{prefix}_downsampling_y_1.npy')
    for i in tqdm(range(2, parts+1)):
        out_x = np.concatenate((out_x, np.load(f'{prefix}_downsampling_x_{i}.npy')))
        out_y = np.concatenate((out_y, np.load(f'{prefix}_downsampling_y_{i}.npy')))
    np.save(f'{prefix}_downsampling_x.npy', out_x.reshape((out_x.shape[0], out_x.shape[2])))
    np.save(f'{prefix}_downsampling_y.npy', out_y.T[0])
