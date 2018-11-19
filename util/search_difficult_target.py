import pickle
from pathlib import Path
import random
import os
import logging
import time
import sys

import numpy as np
from sklearn import metrics
from tqdm import tqdm
from Bio.Blast.Applications import NcbipsiblastCommandline, NcbideltablastCommandline
from Bio.Application import ApplicationError
from Bio import SeqIO, SearchIO
from Bio.SCOP import Scop

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


def blast(algo):
    seqindex = SeqIO.index('data/astral-scopdom-seqres-gd-sel-gs-bib-40-1.75.fa',
                           'fasta', key_function=lambda x: x.split()[0])
    hie = pickle.load(Path('data/train/scop40_1fold_hie.pkl').open('rb'))
    scop = Scop(dir_path='data/scop', version='1.75')
    tmpdir = Path(f'.{algo}')
    tmpdir.mkdir(exist_ok=True)
    auc_result = {}
    for sf in tqdm(hie):
        px_list = hie[sf]
        if len(px_list) < 1:
            continue
        sid = random.sample(px_list, 1)[0]
        record = seqindex[sid]
        f_fasta = tmpdir/f'{sid}.fasta'
        f_xml = tmpdir/f'{sid}.xml'
        SeqIO.write(record, f_fasta.as_posix(), 'fasta')
        try:
            if algo == 'psiblast':
                NcbipsiblastCommandline(query=f_fasta.as_posix(),
                                        db='astral-scopdom-seqres-gd-sel-gs-bib-40-1.75',
                                        num_threads=int(os.cpu_count()),
                                        num_iterations=3,
                                        evalue=999999,
                                        outfmt=5,
                                        out=f_xml.as_posix())()
            elif algo == 'deltablast':
                NcbideltablastCommandline(query=f_fasta.as_posix(),
                                          db='astral-scopdom-seqres-gd-sel-gs-bib-40-1.75',
                                          num_threads=int(os.cpu_count()),
                                          num_iterations=3,
                                          evalue=999999,
                                          outfmt=5,
                                          out=f_xml.as_posix())()
            else:
                raise ValueError(f'Invalid algorithm ({algo})')
        except ApplicationError as e:
            logging.error(e)
            f_xml.unlink()
            continue
        finally:
            f_fasta.unlink()
        results = SearchIO.parse(f_xml.as_posix(), 'blast-xml')
        results = list(results)[-1]
        results = list(results)[:500]
        sf_sccs = scop.getNodeBySunid(sf).sccs
        roc_score = []
        roc_label = []
        for result in results:
            result_sf_sccs = result.description.split(' ')[0][:-2]
            roc_score.append(-result.hsps[0].evalue)
            if result_sf_sccs == sf_sccs:
                roc_label.append(1)
            else:
                roc_label.append(0)
        if np.all(np.array(roc_label) == 1):
            auc = 1.0
        elif np.all(np.array(roc_label) == 0):
            auc = 0.0
        else:
            auc = metrics.roc_auc_score(roc_label, roc_score)
        auc_result[sf_sccs] = {'auc': auc, 'sample': sid, 'num': len(results)}
        f_xml.unlink()
    now = int(time.time())
    pickle.dump(auc_result, Path(f'auc_result_{algo}_{now}.pkl').open('wb'))


def main():
    if sys.argv[1] == 'psiblast':
        blast('psiblast')
    elif sys.argv[1] == 'deltablast':
        blast('deltablast')
    else:
        raise ValueError(f'Invalid algorithm ({algo})')


if __name__ == '__main__':
    main()
