from pathlib import Path
import logging
import itertools
import re
import os

import joblib
from Bio.Align.AlignInfo import PSSM
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import pyflann


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

AA = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
# ToDo: Assert AA order in original PSSM file
WINDOW_WIDTH = 5
WINDOW_CENTER = int(WINDOW_WIDTH / 2)
USE_PADDING_LABEL = False
LENGTH_RATIO = 1/2


def get_feature_vector(pssm1, pssm2, pos1, pos2):
    vec1, vec2 = [], []
    margin_vec1, margin_vec2 = False, False
    for i in range(WINDOW_WIDTH):
        if pos1-WINDOW_CENTER+i < 0 or pos1-WINDOW_CENTER+i >= len(pssm1.pssm):
            margin_vec1 = True
            vec1 += [0] * len(AA)
        else:
            vec1 += [pssm1[pos1 - WINDOW_CENTER + i][x] for x in AA]

        if pos2-WINDOW_CENTER+i < 0 or pos2-WINDOW_CENTER+i >= len(pssm2.pssm):
            margin_vec2 = True
            vec2 += [0] * len(AA)
        else:
            vec2 += [pssm2[pos2 - WINDOW_CENTER + i][x] for x in AA]

    if USE_PADDING_LABEL:
        if margin_vec1:
            vec1.append(1)
        else:
            vec1.append(0)
        if margin_vec2:
            vec2.append(1)
        else:
            vec2.append(0)

    return np.array(vec1), np.array(vec2)


def parse_pssm(fname):
    pssm = []
    with open(fname) as f:
        for line in f.readlines():
            token = line.rstrip('\r\n').split()
            if len(token) == 0:
                continue
            if re.match(r'\d+', token[0]):
                pssm.append((token[1], dict((x, int(y)) for x, y in zip(AA, token[2:22]))))
    return PSSM(pssm)


def get_labels(msa):
    labels = np.zeros((len(msa[0].seq.ungap('-')), len(msa[1].seq.ungap('-'))))
    x, y = 0, 0
    for i in range(msa.get_alignment_length()):
        if msa[0][i] == "-":
            y += 1
        elif msa[1][i] == "-":
            x += 1
        else:
            labels[x, y] = 1
            x += 1
            y += 1

    return labels


def _get_test_vector_set(pssm1: PSSM, pssm2: PSSM, n_features=202):
    # Explicitly count the number of feature vectors
    i = 0
    for x1, x2 in itertools.product(range(len(pssm1.pssm)), range(len(pssm2.pssm))):
        if x1 > int(len(pssm1.pssm)*LENGTH_RATIO) + x2 or x2 > int(len(pssm2.pssm)*LENGTH_RATIO) + x1:
            continue
        i += 1
    tests = np.empty((i, n_features))
    i = 0
    for x1, x2 in itertools.product(range(len(pssm1.pssm)), range(len(pssm2.pssm))):
        if x1 > int(len(pssm1.pssm)*LENGTH_RATIO) + x2 or x2 > int(len(pssm2.pssm)*LENGTH_RATIO) + x1:
            continue
        vec1, vec2 = get_feature_vector(pssm1, pssm2, x1, x2)
        tests[i, :] = np.array(list(vec1) + list(vec2))
        i += 1
    return tests


def predict_by_kmknc(x, y, dim, model_name, args):  # args = [(px1, px2), ...]
    global WINDOW_WIDTH
    global WINDOW_CENTER
    WINDOW_WIDTH = 5
    WINDOW_CENTER = int(WINDOW_WIDTH / 2)
    max_n_neighbors = 100
    kmeans_model = joblib.load(f'data/train/{model_name}.pkl')
    kmeans_model.verbose = False
    for arg in tqdm(args):
        px1, px2 = arg[0], arg[1]
        pred_dir = Path(f'data/prediction/{model_name}/{px1}')
        fname = pred_dir/f'{px2}.npy'
        if fname.exists() or px2.find('.') > -1:
            continue
        pred_dir.mkdir(exist_ok=True, parents=True)
        pssm1 = parse_pssm(f'data/pssm/{px1[2:4]}/{px1}.mtx')
        pssm2 = parse_pssm(f'data/pssm/{px2[2:4]}/{px2}.mtx')
        tests = _get_test_vector_set(pssm1, pssm2, dim)
        labels = kmeans_model.predict(tests)
        proba = np.empty((labels.shape[0], 2))
        for l in np.unique(labels):
            x_ = x[np.where(kmeans_model.labels_ == l)]
            y_ = y[np.where(kmeans_model.labels_ == l)]
            n_neighbors = x_.shape[0]
            if n_neighbors > max_n_neighbors:
                n_neighbors = max_n_neighbors
            knc_model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=os.cpu_count()).fit(x_, y_)
            proba[np.where(labels == l), :] = knc_model.predict_proba(tests[np.where(labels == l)])
        proba = proba[:, 1].reshape((len(pssm1.pssm), len(pssm2.pssm)))
        np.save(fname, proba)


def predict_by_flann_old(x, y, dim, model_name, num_neighbors, args):  # args = [(px1, px2), ...]
    global WINDOW_WIDTH
    global WINDOW_CENTER
    WINDOW_WIDTH = 5
    WINDOW_CENTER = int(WINDOW_WIDTH / 2)
    x = x.astype(np.int32)
    model = pyflann.FLANN()
    model.load_index(f'data/train/{model_name}', x)
    for a in tqdm(args):
        query, template = a[0], a[1]
        pred_dir = Path(f'data/prediction/{model_name}_nn{num_neighbors}/{query}')
        fname = pred_dir/f'{template}.npy'
        if fname.exists() or template.find('.') > -1:
            continue
        pred_dir.mkdir(exist_ok=True, parents=True)
        pssm1 = parse_pssm(f'data/pssm/{query[2:4]}/{query}.mtx')
        pssm2 = parse_pssm(f'data/pssm/{template[2:4]}/{template}.mtx')
        samples = _get_test_vector_set(pssm1, pssm2, dim).astype(np.int32)
        result, _ = model.nn_index(samples, num_neighbors=num_neighbors)
        proba = np.array([np.count_nonzero(y[_])/num_neighbors for _ in result]).reshape((len(pssm1.pssm), len(pssm2.pssm)))
        np.save(fname, proba)


def predict_by_flann_batch(x_path: str, y_path: str, model_path: str, num_neighbors: int, out_dir: str, pssm_dir: str, args):
    x = np.load(x_path).astype(np.int32)
    y = np.load(y_path)
    model = pyflann.FLANN()
    model.load_index(model_path, x)
    for a in tqdm(args):
        query, template = a[0], a[1]
        out_query_dir = Path(out_dir)/query
        fname = out_query_dir/f'{template}.npy'
        if fname.exists():
            continue
        out_query_dir.mkdir(exist_ok=True, parents=True)
        pssm1 = parse_pssm(f'{pssm_dir}/{query[2:4]}/{query}.mtx')
        pssm2 = parse_pssm(f'{pssm_dir}/{template[2:4]}/{template}.mtx')
        samples = _get_test_vector_set(pssm1, pssm2, x.shape[1]).astype(np.int32)
        result, _ = model.nn_index(samples, num_neighbors=num_neighbors)
        proba = np.full((len(pssm1.pssm), len(pssm2.pssm)), -1.0)
        i = 0
        for x1, x2 in itertools.product(range(len(pssm1.pssm)), range(len(pssm2.pssm))):
            if x1 > int(len(pssm1.pssm)*LENGTH_RATIO) + x2 or x2 > int(len(pssm2.pssm)*LENGTH_RATIO) + x1:
                continue
            proba[x1, x2] = np.count_nonzero(y[result[i]]) / num_neighbors
            i += 1
        np.save(fname, proba)


def predict_by_flann(x_path: Path, y_path: Path, model_path: Path, num_neighbors: int, out_path: Path, query: Path, template: Path):
    x = np.load(x_path).astype(np.int32)
    y = np.load(y_path)
    model = pyflann.FLANN()
    model.load_index(model_path.as_posix(), x)
    pssm1 = parse_pssm(query)
    pssm2 = parse_pssm(template)
    samples = _get_test_vector_set(pssm1, pssm2, x.shape[1]).astype(np.int32)
    result, _ = model.nn_index(samples, num_neighbors=num_neighbors)
    proba = np.full((len(pssm1.pssm), len(pssm2.pssm)), -1.0)
    i = 0
    for x1, x2 in itertools.product(range(len(pssm1.pssm)), range(len(pssm2.pssm))):
        if x1 > int(len(pssm1.pssm) * LENGTH_RATIO) + x2 or x2 > int(len(pssm2.pssm) * LENGTH_RATIO) + x1:
            continue
        proba[x1, x2] = np.count_nonzero(y[result[i]]) / num_neighbors
        i += 1
    out_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(out_path.as_posix(), proba)


def predict_by_flann_batch(x, y, model, num_neighbors: int, out_path: Path, query: Path, template: Path):
    pssm1 = parse_pssm(query)
    pssm2 = parse_pssm(template)
    samples = _get_test_vector_set(pssm1, pssm2, x.shape[1]).astype(np.int32)
    result, _ = model.nn_index(samples, num_neighbors=num_neighbors)
    proba = np.full((len(pssm1.pssm), len(pssm2.pssm)), -1.0)
    i = 0
    for x1, x2 in itertools.product(range(len(pssm1.pssm)), range(len(pssm2.pssm))):
        if x1 > int(len(pssm1.pssm) * LENGTH_RATIO) + x2 or x2 > int(len(pssm2.pssm) * LENGTH_RATIO) + x1:
            continue
        proba[x1, x2] = np.count_nonzero(y[result[i]]) / num_neighbors
        i += 1
    out_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(out_path.as_posix(), proba)


def predict_scores(query: Path, template: Path,
                   flann_x=Path('scop40_logscore_tmscore0.5_window5_ratio0.1_x.npy'),
                   flann_y=Path('scop40_logscore_tmscore0.5_window5_ratio0.1_y.npy'),
                   flann_index=Path('flann19_scop40_logscore_tmscore0.5_window5_ratio0.1'),
                   num_neighbors=1000,
                   out_dir=Path('results'),
                   out_name=Path('score.npy')):
    out_path = out_dir / out_name
    if out_path.exists():
        logging.warning(f'Result already exists: {out_path}')
        logging.warning('Do nothing')
    else:
        predict_by_flann(flann_x, flann_y, flann_index, num_neighbors, out_path, query, template)
        logging.info(f'Result is saved into: {out_path}')


def predict_scores_batch(query: Path, template: Path,
                         x, y, model,
                         num_neighbors=1000,
                         out_dir=Path('results'),
                         out_name=Path('score.npy')):
    out_path = out_dir / out_name
    if out_path.exists():
        logging.warning(f'Result already exists: {out_path}')
        logging.warning('Do nothing')
    else:
        predict_by_flann_batch(x, y, model, num_neighbors, out_path, query, template)
        logging.info(f'Result is saved into: {out_path}')
