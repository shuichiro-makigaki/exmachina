import os
import itertools
from pathlib import Path
import logging
import multiprocessing
import queue

from Bio.SCOP import Scop
from Bio import SeqIO

from TMtoolsCommandLine import TMalignCommandLine

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

SCOP_VER = '1.75'
SCOP_DIR = Path('data/scop')
ASTRAL_DIR = Path('data/scop_e')


def run_tmalign_async(q, i):
    f = open(f'data/scop40_structural_alignment_part_{i}.fasta', 'w')
    dbindex = SeqIO.index('data/scop40_structural_alignment.fasta', 'fasta')
    scop40seqres = SeqIO.index('data/astral-scopdom-seqres-gd-sel-gs-bib-40-1.75.fa', 'fasta')
    while True:
        try:
            m = q.get(timeout=60)
        except queue.Empty:
            logging.info('queue.Empty')
            break
        id1, id2 = Path(m[0]).stem, Path(m[1]).stem
        if f'{id1}&{id2}' in dbindex and f'{id2}&{id1}' in dbindex:
            continue
        if id1 in scop40seqres and id2 in scop40seqres:
            try:
                tmalign = TMalignCommandLine(m[0], m[1])
                tmalign.run()
                SeqIO.write(tmalign.alignment, f, 'fasta')
                f.flush()
            except Exception as e:
                logging.error(m)
                logging.exception(e)
                continue
    f.close()


def enqueue_pair_list(q, test=False):
    all_count = 0
    logging.info('Loading SCOP database...')
    scop_root = Scop(dir_path=SCOP_DIR, version=SCOP_VER).getRoot()
    logging.info('Generating training data...')
    for cl in scop_root.getChildren():
        for cf in cl.getChildren():
            for sf in cf.getChildren():
                px = sf.getDescendents('px')
                if len(px) < 2:  # Skip only one member superfamily
                    continue
                for c in itertools.combinations(px, 2):
                    q.put((ASTRAL_DIR/c[0].sid[2:4]/f'{c[0].sid}.ent', ASTRAL_DIR/c[1].sid[2:4]/f'{c[1].sid}.ent'))
                    all_count += 1
    logging.info(f'Number of training data: {all_count}')


def main():
    m = multiprocessing.Manager()
    q = m.Queue()
    pool = [multiprocessing.Process(target=run_tmalign_async, args=(q, i))
            for i in range(os.cpu_count() / 2)]
    logging.info('Running TMalign...')
    [p.start() for p in pool]
    enqueue_pair_list(q)
    [p.join() for p in pool]
    logging.info('Done')


if __name__ == '__main__':
    main()
