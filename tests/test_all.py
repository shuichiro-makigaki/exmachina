from unittest import TestCase

import numpy as np

import machina.generate_training_dataset
from machina.generate_training_dataset import \
    create_dataset, parse_pssm, parse_alignment, scan_rectangle, get_feature_vector
from machina.generate_alignment import _pp

import os
if not os.getenv('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


machina.generate_training_dataset.WINDOW_WIDTH = 5
machina.generate_training_dataset.WINDOW_CENTER = int(5 / 2)
machina.generate_training_dataset.USE_PADDING_LABEL = True
WINDOW_CENTER = machina.generate_training_dataset.WINDOW_CENTER


class TestGenerateAlignment(TestCase):
    def test__pp(self):
        seq = _pp('tests/data/pssm/1dlwa_.mtx')
        assert len(seq) == 116
        assert str(seq)[:5] == 'SLFEQ'
        assert str(seq)[-5:] == 'DVVTV'
        seq = _pp('tests/data/pssm/7hbia_.mtx')
        assert len(seq) == 145
        assert str(seq)[:5] == 'SVYDA'
        assert str(seq)[-5:] == 'VQAAL'


class TestParsePSSM(TestCase):
    def test_parse_pssm(self):
        assert parse_pssm('tests/data/pssm/1dlwa_.mtx')
        assert parse_pssm('tests/data/pssm/7hbia_.mtx')


class TestParseMSA(TestCase):
    def test_parse_alignment(self):
        assert parse_alignment('d1dlwa_', 'd7hbia_', db_file='tests/data/scop40_structural_alignment.fasta')


class TestScan(TestCase):
    def setUp(self):
        self.pssm1 = parse_pssm('tests/data/pssm/1dlwa_.mtx')
        self.pssm2 = parse_pssm('tests/data/pssm/7hbia_.mtx')
        self.msa = parse_alignment('d1dlwa_', 'd7hbia_', db_file='tests/data/scop40_structural_alignment.fasta')

    def test_create_dataset(self):
        vec, win = create_dataset(self.pssm1, self.pssm2, self.msa)
        # print(win)
        assert win[14][2][2] == 1
        assert win[14][3][4] == 1
        assert win[15][1][1] == 1
        assert win[15][2][2] == 0
        assert win[15][2][3] == 1
        assert win[15][3][4] == 1
        assert win[15][4].all() == 0
        assert win[16][2][2] == 1
        assert win[16][3][3] == 1
        assert win[16][4][4] == 1
        assert win[16][1][0] == 1
        assert win[16][0].all() == 0
        assert np.array(vec[0][0]).all() is None
        assert np.array(vec[0][1]).all() is None
        for i in range(self.msa.get_alignment_length()):
            if self.msa[0][i] == "-":
                assert win[i][WINDOW_CENTER][WINDOW_CENTER] == 0
            elif self.msa[1][i] == "-":
                assert win[i][WINDOW_CENTER][WINDOW_CENTER] == 0
            else:
                assert win[i][WINDOW_CENTER][WINDOW_CENTER] == 1

    def test_scan_rectangle(self):
        dirty_map = np.zeros((len(self.pssm1.pssm), len(self.pssm2.pssm)))
        vec, window = scan_rectangle(self.pssm1, self.pssm2, 0, 0, self.msa, 0, dirty_map)
        assert window.all() == 0
        assert dirty_map[:3, :3].all() == 1
        assert vec[2][2][0][:2, :60].all() == 0
        assert vec[2][2][1] == 0
        # assert vec[:2, :].all() is None and vec[:, :2].all() is None
        vec, window = scan_rectangle(self.pssm1, self.pssm2, 0, 14, self.msa, 14, dirty_map)
        assert window[2, 2] == 1 and window[3, 4] == 1
        assert dirty_map[0:3, 12:17].all() == 1
        # assert vec[:2, :].all() is None
        vec, window = scan_rectangle(self.pssm1, self.pssm2, len(self.pssm1.pssm)-3, len(self.pssm2.pssm)-1,
                                     self.msa, 148, dirty_map)
        assert window[0, 1] == 1 and window[2, 2] == 1
        assert dirty_map[-5:, -3:].all() == 1
        # assert vec[:, -2:].all() is None

    def test_get_window_vector(self):
        vec = get_feature_vector(self.pssm1, self.pssm2, 0, 0)
        assert len(vec[0]) == 101 and len(vec[1]) == 101
        assert vec[0][-1] == 1 and vec[1][-1] == 1
        assert vec[0][:40].all() == 0 and vec[1][:40].all() == 0
