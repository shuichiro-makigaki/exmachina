import logging

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np
import pyflann

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


def train_kmeans(x, model_path, n_clusters):
    logging.info(f'Training x.shape={x.shape}...')
    kmeans_model = MiniBatchKMeans(n_clusters=n_clusters).fit(x)
    logging.info(f'Saving model...')
    joblib.dump(kmeans_model, model_path)


def train_knc(x, y, model_path, n_neighbors):
    logging.info(f'Training (X.shape, Y.shape) = ({x.shape}, {y.shape})...')
    knc_model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1).fit(x, y)
    joblib.dump(knc_model, model_path)


def train_rfc(x, y, model_path, n_estimators):
    logging.info(f'Training (X.shape, Y.shape) = ({x.shape}, {y.shape})...')
    rfc_model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1).fit(x, y)
    joblib.dump(rfc_model, model_path)


def train_flann(x, model_path):
    samples = x.astype(np.int32)
    logging.info(f'Indexing samples.shape = {samples.shape}...')
    model = pyflann.FLANN()
    model.build_index(samples)
    logging.info(f'Saving index...')
    model.save_index(model_path)
