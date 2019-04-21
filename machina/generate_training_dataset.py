import re
import logging
from pathlib import Path
import itertools
import csv
import pickle
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import shutil
import os
import random

from Bio import SeqIO
from Bio.Alphabet import generic_protein
from Bio.Align import MultipleSeqAlignment
from Bio.Align.AlignInfo import PSSM
import numpy as np
from tqdm import tqdm
import more_itertools

from machina.TMtoolsCommandLine import TMalignCommandLine

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

AA = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
# ToDo: Assert AA order in original PSSM file
TMSCORE_THRESHOLD = 0.5
WINDOW_WIDTH = 5
WINDOW_CENTER = int(WINDOW_WIDTH / 2)
USE_PADDING_LABEL = False
PART_DIR = 'data/train/process_part'


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


def parse_alignment(id1, id2, db_index=None, db_file='data/train/scop40_structural_alignment.fasta'):
    if db_index is None:
        db_index = SeqIO.index(db_file, 'fasta')
    return MultipleSeqAlignment([db_index[f'{id1}&{id2}'], db_index[f'{id2}&{id1}']], alphabet=generic_protein)


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


def scan_rectangle(pssm1, pssm2, pos1, pos2, msa, pos_msa, dirty_map):
    window = np.zeros((WINDOW_WIDTH, WINDOW_WIDTH))
    # Center
    if '-' not in msa[:, pos_msa]:
        window[WINDOW_CENTER, WINDOW_CENTER] = 1
    # Backward
    x, y = WINDOW_CENTER, WINDOW_CENTER
    for i in range(1, WINDOW_CENTER+1):
        try:
            sub = msa[:, pos_msa-i]
        except IndexError:
            continue
        if '-' not in sub:
            x -= 1
            y -= 1
            window[x, y] = 1
        elif sub.startswith('-'):
            y -= 1
        elif sub.endswith('-'):
            x -= 1
    # Forward
    x, y = WINDOW_CENTER, WINDOW_CENTER
    for i in range(0, WINDOW_CENTER+1):
        try:
            sub = msa[:, pos_msa+i]
        except IndexError:
            continue
        if '-' not in sub:
            window[x, y] = 1
            x += 1
            y += 1
        elif sub.startswith('-'):
            y += 1
        elif sub.endswith('-'):
            x += 1
    # Create vector set in a rectangle
    rec = [[None for x in range(WINDOW_WIDTH)] for y in range(WINDOW_WIDTH)]
    for x, y in itertools.product(range(WINDOW_WIDTH), range(WINDOW_WIDTH)):
        if pos1-WINDOW_CENTER+x < 0 or pos2-WINDOW_CENTER+y < 0:
            continue
        if pos1-WINDOW_CENTER+x >= len(pssm1.pssm) or pos2-WINDOW_CENTER+y >= len(pssm2.pssm):
            continue
        if dirty_map[pos1-WINDOW_CENTER+x, pos2-WINDOW_CENTER+y] == 0:
            v = get_feature_vector(pssm1, pssm2, pos1-WINDOW_CENTER+x, pos2-WINDOW_CENTER+y)
            rec[x][y] = (np.array([v[0], v[1]]), window[x, y])
            dirty_map[pos1-WINDOW_CENTER+x, pos2-WINDOW_CENTER+y] = 1
    return rec, window


def create_dataset(pssm1, pssm2, msa):
    dirty_map = np.zeros((len(pssm1.pssm), len(pssm2.pssm)))
    all_rec, all_window = [], []
    x, y = 0, 0
    for i in range(msa.get_alignment_length()):
        rec, window = scan_rectangle(pssm1, pssm2, x, y, msa, i, dirty_map)
        all_rec.append(rec)
        all_window.append(window)
        if msa[0][i] == "-":
            y += 1
        elif msa[1][i] == "-":
            x += 1
        else:
            x += 1
            y += 1
    return all_rec, all_window


def process(*args):
    pid = multiprocessing.current_process().pid
    db_index = SeqIO.index('data/train/scop40_structural_alignment.fasta', 'fasta')
    with Path(f'{PART_DIR}/pid{pid}.csv').open('a', newline='') as f:
        writer = csv.writer(f)
        for c in args:
            try:
                tmalign = TMalignCommandLine(f'data/scop_e/{c[0][2:4]}/{c[0]}.ent',
                                             f'data/scop_e/{c[1][2:4]}/{c[1]}.ent')
                tmalign.run()
                if max(tmalign.tmscore) < TMSCORE_THRESHOLD:
                    continue
                pssm1 = parse_pssm(f'data/pssm/{c[0][2:4]}/{c[0]}.mtx')
                pssm2 = parse_pssm(f'data/pssm/{c[1][2:4]}/{c[1]}.mtx')
                msa = parse_alignment(c[0], c[1], db_index=db_index)
                assert len(pssm1.pssm) == len(msa[0].seq.ungap('-'))
                assert len(pssm2.pssm) == len(msa[1].seq.ungap('-'))
                recs, _ = create_dataset(pssm1, pssm2, msa)
            except Exception as e:
                logging.error(c)
                logging.exception(e)
                continue
            for r in recs:
                for x, y in itertools.product(range(WINDOW_WIDTH), range(WINDOW_WIDTH)):
                    v = r[x][y]
                    if v:
                        writer.writerow(list(v[0].reshape(WINDOW_WIDTH*20*2))+list([v[1]]))


def create_scop40_dataset_old(window_size, tmscore_threshold, n_jobs=None):
    global WINDOW_WIDTH
    global WINDOW_CENTER
    global TMSCORE_THRESHOLD
    WINDOW_WIDTH = window_size
    TMSCORE_THRESHOLD = tmscore_threshold
    WINDOW_CENTER = int(WINDOW_WIDTH / 2)
    if n_jobs is None:
        n_jobs = os.cpu_count()
    part_dir = Path(PART_DIR)
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir()
    with Path(f'data/train/scop40_hie.pkl').open('rb') as f:
        hie = pickle.load(f)
    for px in tqdm(hie):
        if len(hie[px]) < 2:
            continue
        with Pool(n_jobs) as pool:
            pool.starmap(process,
                         more_itertools.chunked(
                             itertools.combinations(hie[px], 2),
                             n_jobs))


def process2(msa: MultipleSeqAlignment, pssm1: PSSM, pssm2: PSSM, ratio: float):
    assert len(pssm1.pssm) == len(msa[0].seq.ungap('-'))
    assert len(pssm2.pssm) == len(msa[1].seq.ungap('-'))
    recs, _ = create_dataset(pssm1, pssm2, msa)
    results = []
    for r in recs:
        for x, y in itertools.product(range(WINDOW_WIDTH), range(WINDOW_WIDTH)):
            v = r[x][y]
            if v:
                if random.random() <= ratio:
                    results.append(list(v[0].reshape(WINDOW_WIDTH*len(AA)*2))+list([v[1]]))
    return np.array(results, dtype=np.int8)


def create_scop40_dataset(structural_alignments_path: str, pssm_dir: str,
                          out_path: str, ratio: float, masked_ids: list):
    aligns = SeqIO.index(structural_alignments_path, 'fasta')
    finished = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for seq_id in tqdm(aligns):
            if float(aligns[seq_id].description.split('=')[1]) < TMSCORE_THRESHOLD:
                continue
            dom1, dom2 = seq_id.split('&')[:2]
            if dom1 in masked_ids or dom2 in masked_ids:
                continue
            if (dom1, dom2) in finished:
                continue
            futures.append(
                executor.submit(process2,
                                MultipleSeqAlignment([aligns[f'{dom1}&{dom2}'], aligns[f'{dom2}&{dom1}']]),
                                parse_pssm(f'{pssm_dir}/{dom1[2:4]}/{dom1}.mtx'),
                                parse_pssm(f'{pssm_dir}/{dom2[2:4]}/{dom2}.mtx'),
                                ratio))
            finished.append((dom1, dom2))
            finished.append((dom2, dom1))
        results = []
        for future in as_completed(futures):
            results.extend(future.result())
        results = np.array(results, dtype=np.int8)
        np.save(out_path, results)
        logging.info(f'Training data shape={results.shape}')


def split_data_label(sample_path: str, out_x_path: str, out_y_path: str):
    samples = np.load(sample_path)
    i = 0
    x = np.empty((samples.shape[0], samples.shape[1]-1), dtype=np.int8)
    y = np.empty(samples.shape[0], dtype=np.int8)
    for v in tqdm(samples):
        x[i, :] = v[:-1]
        y[i] = v[-1]
        i += 1
    np.save(out_x_path, x)
    np.save(out_y_path, y)


def get_validation_label(domain_sid1: str, domain_sid2: str, aligns, pssm_dir: str):
    pssm1 = parse_pssm(f'{pssm_dir}/{domain_sid1[2:4]}/{domain_sid1}.mtx')
    pssm2 = parse_pssm(f'{pssm_dir}/{domain_sid2[2:4]}/{domain_sid2}.mtx')
    msa = MultipleSeqAlignment([aligns[f'{domain_sid1}&{domain_sid2}'], aligns[f'{domain_sid2}&{domain_sid1}']])
    assert len(pssm1.pssm) == len(msa[0].seq.ungap('-')) and len(pssm2.pssm) == len(msa[1].seq.ungap('-'))
    Y = np.zeros((len(pssm1.pssm), len(pssm2.pssm)), dtype=np.int8)
    x, y = 0, 0
    for i in range(msa.get_alignment_length()):
        if msa[0][i] == "-":
            y += 1
        elif msa[1][i] == "-":
            x += 1
        else:
            Y[x, y] = 1
            x += 1
            y += 1

    return Y
