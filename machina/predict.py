from pathlib import Path
import logging
import itertools
import re
import os
from concurrent.futures import ProcessPoolExecutor
import more_itertools

from sklearn.externals import joblib
from Bio.Align.AlignInfo import PSSM
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import pyflann
# from keras.models import load_model


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

AA = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
# ToDo: Assert AA order in original PSSM file
WINDOW_WIDTH = 9
WINDOW_CENTER = int(WINDOW_WIDTH / 2)
USE_PADDING_LABEL = False
# MODEL_NAME = 'NN_scop40_tmscore50_w9_downsampling_ep20_ba512'
# MODEL_NAME = 'NN_scop40_tmscore50_w9_downsampling_ep20_ba1024'
MODEL_NAME = 'NN_scop40_tmscore50_w9_downsampling_ep20_ba2048'


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
    tests = np.empty((len(pssm1.pssm)*len(pssm2.pssm), n_features))
    i = 0
    for x1, x2 in itertools.product(range(len(pssm1.pssm)), range(len(pssm2.pssm))):
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


def predict_by_flann(x, y, dim, model_name, num_neighbors, args):  # args = [(px1, px2), ...]
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


def _predict_by_rfc(args):  # args = [(px1, px2), ...]
    Path(f'data/prediction/{MODEL_NAME}').mkdir(exist_ok=True)
    model_file = Path(f'data/models/{MODEL_NAME}.pkl')
    model = joblib.load(model_file)
    model.verbose = False
    for arg in args:
        px1, px2 = arg[0], arg[1]
        pred_dir = Path(f'data/prediction/{MODEL_NAME}/{px1}')
        pred_dir.mkdir(exist_ok=True)
        fname = pred_dir/f'{px2}.npy'
        if fname.exists():
            logging.debug(f'{fname} already exists. Skipping.')
            continue
        if px2.find('.') > -1:
            logging.debug(f'Ignore {px2}')
            continue
        try:
            pssm1 = parse_pssm(f'data/pssm/{px1[2:4]}/{px1}.mtx')
            pssm2 = parse_pssm(f'data/pssm/{px2[2:4]}/{px2}.mtx')
            tests = _get_test_vector_set(pssm1, pssm2)
            proba = model.predict_proba(tests)[:, 1].reshape((len(pssm1.pssm), len(pssm2.pssm)))
        except Exception as e:
            logging.error(arg)
            logging.exception(e)
            continue
        np.save(fname, proba)


def predict_by_rfc(n_jobs, args):  # args = [(px1, px2), ...]
    args = more_itertools.chunked(args, int(len(args)/n_jobs))
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        [_ for _ in list(executor.map(_predict_by_rfc, args))]


def _predict_by_nn(args):  # args = [(px1, px2), ...]
    Path(f'data/prediction/{MODEL_NAME}').mkdir(exist_ok=True)
    model = load_model(Path(f'data/models/{MODEL_NAME}.h5'))
    for arg in tqdm(args):
        px1, px2 = arg[0], arg[1]
        pred_dir = Path(f'data/prediction/{MODEL_NAME}/{px1}')
        pred_dir.mkdir(exist_ok=True)
        fname = pred_dir/f'{px2}.npy'
        if fname.exists():
            logging.debug(f'{fname} already exists. Skipping.')
            continue
        if px2.find('.') > -1:
            logging.debug(f'Ignore {px2}')
            continue
        try:
            pssm1 = parse_pssm(f'data/pssm/{px1[2:4]}/{px1}.mtx')
            pssm2 = parse_pssm(f'data/pssm/{px2[2:4]}/{px2}.mtx')
            tests = _get_test_vector_set(pssm1, pssm2, 360)
            proba = model.predict(tests)
            proba = proba[:, 0].reshape((len(pssm1.pssm), len(pssm2.pssm)))
        except Exception as e:
            logging.error(arg)
            logging.exception(e)
            continue
        np.save(fname, proba)


def predict_by_nn(n_jobs, args):  # args = [(px1, px2), ...]
    args = more_itertools.chunked(args, int(len(args)/n_jobs))
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        [_ for _ in list(executor.map(_predict_by_nn, args))]
