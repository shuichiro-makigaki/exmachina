import logging
from pathlib import Path
import re

import numpy as np
from Bio.Seq import Seq
from Bio.Alphabet import generic_protein
from Bio.SubsMat import MatrixInfo

from machina.pairwise2 import align

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

BLOSUM_MODE = 'off'


class _my_match(object):
    def __init__(self, matrix):
        """Initialize the class."""
        self.match = matrix

    def __call__(self, charA, charB, posA, posB):
        """Call a match function instance already created."""
        if BLOSUM_MODE == 'on':
            if (charA, charB) in MatrixInfo.blosum62:
                return self.match[posA][posB] + MatrixInfo.blosum62[(charA, charB)]
            else:
                return self.match[posA][posB] + MatrixInfo.blosum62[(charB, charA)]
        elif BLOSUM_MODE == 'only':
            if (charA, charB) in MatrixInfo.blosum62:
                return MatrixInfo.blosum62[(charA, charB)]
            else:
                return MatrixInfo.blosum62[(charB, charA)]
        else:
            return self.match[posA][posB]


def _pp(path):
    seq = []
    for line in Path(path).read_text().splitlines():
        token = line.rstrip('\r\n').split()
        if len(token) == 0:
            continue
        if re.match(r'\d+', token[0]):
            seq.append(token[1])
    return Seq(''.join(seq), generic_protein)


def local_alignment(path, gapopen, gapextend):
    domain_id = path.stem
    seq_b = str(_pp(f'data/pssm/{domain_id[2:4]}/{domain_id}.mtx'))
    domain_id = path.parts[-2]
    seq_a = str(_pp(f'data/pssm/{domain_id[2:4]}/{domain_id}.mtx'))
    try:
        matrix = np.load(path.as_posix())
        ali = align.localcs(seq_a, seq_b, _my_match(matrix.tolist()), gapopen, gapextend, force_generic=True)
    except Exception as e:
        logging.error(" ")
        logging.exception(e)
        logging.error(path)
        logging.error(seq_a)
        logging.error(seq_b)
        ali = None
    return ali


def generate_alignment(path, gap_open, gap_extend):
    ali = local_alignment(path, gapopen=gap_open, gapextend=gap_extend)
    return {path.stem: ali}
