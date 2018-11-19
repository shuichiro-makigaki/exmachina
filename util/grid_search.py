from pathlib import Path
import logging
import os
import pickle
import datetime

import numpy as np
from sklearn import metrics
from Bio.SCOP import Scop
from Bio.Blast.Applications import NcbideltablastCommandline, NcbipsiblastCommandline
from Bio import SeqIO, SearchIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_protein
import matplotlib.pyplot
import matplotlib.figure
import matplotlib.axes
import pandas
import seaborn as sns
from tqdm import tqdm

hie = pickle.load(Path('data/train/scop40_1fold_hie.pkl').open('rb'))
scop100 = Scop(dir_path='data/scop', version='1.75')
seqindex = SeqIO.index('data/astral-scopdom-seqres-gd-sel-gs-bib-40-1.75.fa', 'fasta')
test_data = (
    ('d1wlqc_', datetime.datetime(2009, 2, 17, 0, 0), 'a.4.5', 762),
    ('d2axtu1', datetime.datetime(2009, 2, 10, 0, 0), 'a.60.12', 159),
    ('d2zqna1', datetime.datetime(2009, 2, 10, 0, 0), 'b.42.2', 119),
    ('d1qg3a1', datetime.datetime(2009, 1, 20, 0, 0), 'b.1.2', 344),
    ('d1wzca1', datetime.datetime(2009, 1, 27, 0, 0), 'c.108.1', 296),
    ('d2dsta1', datetime.datetime(2009, 1, 27, 0, 0), 'c.69.1', 975),
    ('d1y5ha3', datetime.datetime(2009, 2, 10, 0, 0), 'd.37.1', 62),
    ('d2pzza1', datetime.datetime(2009, 1, 20, 0, 0), 'd.77.1', 92),
    ('d1ni9a_', datetime.datetime(2009, 2, 10, 0, 0), 'e.7.1', 151),
    ('d3cw9a1', datetime.datetime(2008, 9, 2, 0, 0), 'e.23.1', 22),
    ('d2axtd1', datetime.datetime(2009, 2, 10, 0, 0), 'f.26.1', 174),
    ('d2axto1', datetime.datetime(2009, 2, 10, 0, 0), 'f.4.1', 15),
    ('d2vy4a1', datetime.datetime(2009, 2, 17, 0, 0), 'g.37.1', 182),
    ('d3d9ta1', datetime.datetime(2009, 2, 10, 0, 0), 'g.52.1', 81),
)


def proposed(query, gap_open, gap_extend, roc_top, result_dict_cover, fold, clst, blosum_flg, kei, lam):
    def _max_score(_x):
        return max(list(_x.values())[0], key=lambda __x: __x[2])[2]
    model_name = f'kmeans_scop40_1fold_random_{fold}fold_clst{clst}_iter100'
    result = np.load(f'data/result_{query}_{model_name}_blosum62_{blosum_flg}_open{gap_open}_ext{gap_extend}.npy')
    rank = sorted(result.tolist(), key=_max_score, reverse=True)
    rank = [{list(x.keys())[0]: list(x.values())[0]} for x in rank]
    roc_score, roc_label = [], []
    all_tp_count = 0
    query_len = len(seqindex[query].seq)
    for r in rank:
        tid = list(r.keys())[0]
        qdom, tdom = scop100.getDomainBySid(query), scop100.getDomainBySid(tid)
        qsccs, tsccs = qdom.getAscendent('sf').sccs, tdom.getAscendent('sf').sccs
        all_tp_count = len(hie[qdom.getAscendent('sf').sunid])
        # score = kei * 10500 * query_len * np.exp(-r[tid][0][2]*lam)
        score = r[tid][0][2]
        roc_score.append(score)
        label = 1 if qsccs == tsccs else 0
        roc_label.append(label)
        if label == 1:
            result_dict_cover['proposed'].append(len(Seq(r[tid][0][0], generic_protein).ungap('-')) / query_len)
        if roc_label.count(0) == roc_top:
            break
    fpc, tpc = [0], [0]
    for i in range(0, roc_top):
        fpc.append(i)
        tpc.append(roc_label[:i+1].count(1))
    auc = metrics.auc(fpc, tpc) / roc_top / all_tp_count
    return fpc, tpc, auc


def proposed2(query, gap_open, gap_extend, roc_top, result_dict_cover,
              blast_ranking, fold, clst, blosum_flg, kei, lam):
    def _max_score(_x):
        return max(list(_x.values())[0], key=lambda __x: __x[2])[2]
    model_name = f'kmeans_scop40_1fold_random_{fold}fold_clst{clst}_iter100'
    result = np.load(f'data/result_{query}_{model_name}_blosum62_{blosum_flg}_open{gap_open}_ext{gap_extend}.npy')
    rank = sorted(result.tolist(), key=_max_score, reverse=True)
    rank = [{list(x.keys())[0]:
                 ('', '', kei * 10500 * len(seqindex[query].seq) * np.exp(-list(x.values())[0][0][2] * lam))}
            for x in rank]
    rank.extend(blast_ranking['DELTA-BLAST_iter1'][query])
    rank = sorted(rank, key=lambda x: list(x.values())[0][2])
    assert len(rank) > 10600
    roc_score, roc_label = [], []
    done = []
    qdom = scop100.getDomainBySid(query)
    qsccs = qdom.getAscendent('sf').sccs
    all_tp_count = len(hie[qdom.getAscendent('sf').sunid])
    for r in rank:
        tid = list(r.keys())[0]
        if '.' in tid or tid in done:
            continue
        else:
            done.append(tid)
        tdom = scop100.getDomainBySid(tid)
        tsccs = tdom.getAscendent('sf').sccs
        score = r[tid][2]
        roc_score.append(score)
        label = 1 if qsccs == tsccs else 0
        roc_label.append(label)
        if label == 1:
            result_dict_cover['proposed2'].append(0)
        if roc_label.count(0) == roc_top:
            break
    fpc, tpc = [0], [0]
    for i in range(0, roc_top):
        fpc.append(i)
        tpc.append(roc_label[:i+1].count(1))
    auc = metrics.auc(fpc, tpc) / roc_top / all_tp_count
    return fpc, tpc, auc


def blast(query, algo, iter, key_name, roc_top, result_dict_cover, blast_ranking):
    tmpdir = Path(f'.{algo}_iter{iter}')
    tmpdir.mkdir(exist_ok=True)
    f_fasta = Path(f'data/test/{query}.fasta')
    f_xml = tmpdir/f'{query}.xml'
    if not f_xml.exists():
        if algo == 'psiblast':
            NcbipsiblastCommandline(query=f_fasta.as_posix(),
                                    db='data/blastdb/astral-scopdom-seqres-gd-sel-gs-bib-40-1.75_train_only',
                                    num_threads=int(os.cpu_count()),
                                    num_iterations=iter,
                                    max_target_seqs=500,
                                    evalue=999999,
                                    outfmt=5,
                                    out=f_xml.as_posix())()
        elif algo == 'deltablast':
            NcbideltablastCommandline(query=f_fasta.as_posix(),
                                      db='data/blastdb/astral-scopdom-seqres-gd-sel-gs-bib-40-1.75_train_only',
                                      rpsdb='data/blastdb/cdd_delta',
                                      num_threads=int(os.cpu_count()),
                                      num_iterations=iter,
                                      max_target_seqs=500,
                                      evalue=999999,
                                      outfmt=5,
                                      out=f_xml.as_posix())()
        elif algo == 'psiblast_pssm':
            f_pssm = Path(f'data/test/{query}.pssm')
            NcbipsiblastCommandline(in_pssm=f_pssm.as_posix(),
                                    db='data/blastdb/astral-scopdom-seqres-gd-sel-gs-bib-40-1.75_train_only',
                                    num_threads=int(os.cpu_count()),
                                    num_iterations=iter,
                                    max_target_seqs=500,
                                    evalue=999999,
                                    outfmt=5,
                                    out=f_xml.as_posix())()
        else:
            raise ValueError(f'Invalid algorithm ({algo})')
    results = SearchIO.parse(f_xml.as_posix(), 'blast-xml')
    results = list(results)[-1]  # final iteration
    results = list(results)
    sf_sccs = scop100.getDomainBySid(query).getAscendent('sf').sccs
    all_tp_count = len(hie[scop100.getDomainBySid(query).getAscendent('sf').sunid])
    roc_score, roc_label = [], []
    assert len(results) >= roc_top
    blast_ranking[key_name][query] = []
    for result in results:
        blast_ranking[key_name][query].append({result.id: ('', '', result.hsps[0].evalue)})
    for result in results:
        result_sf_sccs = '.'.join(result.description.split(' ')[0].split('.')[:-1])
        roc_score.append(-result.hsps[0].evalue)
        label = 1 if result_sf_sccs == sf_sccs else 0
        roc_label.append(label)
        if label == 1:
            result_dict_cover[key_name].append(
                len(result.hsps[0].query.seq.ungap('-')) / len(SeqIO.read(f_fasta.as_posix(), 'fasta').seq))
        if roc_label.count(0) == roc_top:
            break
    fpc, tpc = [0], [0]
    for i in range(0, roc_top):
        fpc.append(i)
        tpc.append(roc_label[:i+1].count(1))
    auc = metrics.auc(fpc, tpc) / roc_top / all_tp_count
    return fpc, tpc, auc


def hmmer(query, algo, iter, key_name, roc_top, result_dict_cover):
    tmpdir = Path(f'.{algo}_iter{iter}')
    tmpdir.mkdir(exist_ok=True)
    f_fasta = tmpdir / f'{query}.fasta'
    f_data = tmpdir / f'{query}.fasta.data'
    results = list(SearchIO.parse(f_data.as_posix(), 'hmmer3-text'))[0]
    results = results.hits
    sf_sccs = scop100.getDomainBySid(query).getAscendent('sf').sccs
    all_tp_count = len(hie[scop100.getDomainBySid(query).getAscendent('sf').sunid])
    roc_score, roc_label = [], []
    for result in results:
        result_sf_sccs = '.'.join(result.description.split(' ')[0].split('.')[:-1])
        roc_score.append(-result.evalue)
        label = 1 if result_sf_sccs == sf_sccs else 0
        roc_label.append(label)
        if roc_label.count(0) == roc_top:
            break
    # Fill False
    while roc_label.count(0) == roc_top:
        roc_label.append(0)
        roc_score.append(999999)
    fpc, tpc = [0], [0]
    for i in range(0, roc_top):
        fpc.append(i)
        tpc.append(roc_label[:i + 1].count(1))
    auc = metrics.auc(fpc, tpc) / roc_top / all_tp_count
    return fpc, tpc, auc


def make_result(query, gap_open, gap_extend, blosum_flg, roc_top, result_dict, result_dict_cover,
                blast_ranking, fold, clst, kei, lam):
    # fig: matplotlib.figure.Figure = None
    # ax: matplotlib.axes.Axes = None
    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 10))

    # PSI-BLAST iter1
    if 'PSI-BLAST_iter1' not in result_dict:
        result_dict['PSI-BLAST_iter1'] = {}
    if 'PSI-BLAST_iter1' not in blast_ranking:
        blast_ranking['PSI-BLAST_iter1'] = {}
    if 'PSI-BLAST_iter1' not in result_dict_cover:
        result_dict_cover['PSI-BLAST_iter1'] = []
    fpr, tpr, auc = blast(query, 'psiblast', 1, 'PSI-BLAST_iter1', roc_top, result_dict_cover, blast_ranking)
    result_dict['PSI-BLAST_iter1'][query] = auc
    ax.plot(fpr, tpr, label=f'PSI-BLAST iter=1 AUC={auc:.4f}')

    # PSI-BLAST iter1 (PSSM)
    if 'PSI-BLAST_PSSM_iter1' not in result_dict:
        result_dict['PSI-BLAST_PSSM_iter1'] = {}
    if 'PSI-BLAST_PSSM_iter1' not in blast_ranking:
        blast_ranking['PSI-BLAST_PSSM_iter1'] = {}
    if 'PSI-BLAST_PSSM_iter1' not in result_dict_cover:
        result_dict_cover['PSI-BLAST_PSSM_iter1'] = []
    fpr, tpr, auc = blast(query, 'psiblast_pssm', 1, 'PSI-BLAST_PSSM_iter1', roc_top, result_dict_cover, blast_ranking)
    result_dict['PSI-BLAST_PSSM_iter1'][query] = auc
    ax.plot(fpr, tpr, label=f'PSI-BLAST_PSSM iter=1 AUC={auc:.4f}')

    # PSI-BLAST iter3
    if 'PSI-BLAST_iter3' not in result_dict:
        result_dict['PSI-BLAST_iter3'] = {}
    if 'PSI-BLAST_iter3' not in blast_ranking:
        blast_ranking['PSI-BLAST_iter3'] = {}
    if 'PSI-BLAST_iter3' not in result_dict_cover:
        result_dict_cover['PSI-BLAST_iter3'] = []
    fpr, tpr, auc = blast(query, 'psiblast', 3, 'PSI-BLAST_iter3', roc_top, result_dict_cover, blast_ranking)
    result_dict['PSI-BLAST_iter3'][query] = auc
    ax.plot(fpr, tpr, label=f'PSI-BLAST iter=3 AUC={auc:.4f}')

    # PSI-BLAST iter5
    if 'PSI-BLAST_iter5' not in result_dict:
        result_dict['PSI-BLAST_iter5'] = {}
    if 'PSI-BLAST_iter5' not in blast_ranking:
        blast_ranking['PSI-BLAST_iter5'] = {}
    if 'PSI-BLAST_iter5' not in result_dict_cover:
        result_dict_cover['PSI-BLAST_iter5'] = []
    fpr, tpr, auc = blast(query, 'psiblast', 5, 'PSI-BLAST_iter5', roc_top, result_dict_cover, blast_ranking)
    result_dict['PSI-BLAST_iter5'][query] = auc
    ax.plot(fpr, tpr, label=f'PSI-BLAST iter=5 AUC={auc:.4f}')

    # DELTA-BLAST iter2
    if 'DELTA-BLAST_iter2' not in result_dict:
        result_dict['DELTA-BLAST_iter2'] = {}
    if 'DELTA-BLAST_iter2' not in blast_ranking:
        blast_ranking['DELTA-BLAST_iter2'] = {}
    if 'DELTA-BLAST_iter2' not in result_dict_cover:
        result_dict_cover['DELTA-BLAST_iter2'] = []
    fpr, tpr, auc = blast(query, 'deltablast', 2, 'DELTA-BLAST_iter2', roc_top, result_dict_cover, blast_ranking)
    result_dict['DELTA-BLAST_iter2'][query] = auc
    ax.plot(fpr, tpr, label=f'DELTA-BLAST iter=2 AUC={auc:.4f}')

    # DELTA-BLAST iter1
    if 'DELTA-BLAST_iter1' not in result_dict:
        result_dict['DELTA-BLAST_iter1'] = {}
    if 'DELTA-BLAST_iter1' not in blast_ranking:
        blast_ranking['DELTA-BLAST_iter1'] = {}
    if 'DELTA-BLAST_iter1' not in result_dict_cover:
        result_dict_cover['DELTA-BLAST_iter1'] = []
    fpr, tpr, auc = blast(query, 'deltablast', 1, 'DELTA-BLAST_iter1', roc_top, result_dict_cover, blast_ranking)
    result_dict['DELTA-BLAST_iter1'][query] = auc
    ax.plot(fpr, tpr, label=f'DELTA-BLAST iter=1 AUC={auc:.4f}')

    # HMMER
    if 'HMMER' not in result_dict:
        result_dict['HMMER'] = {}
    if 'HMMER' not in blast_ranking:
        blast_ranking['HMMER'] = {}
    if 'HMMER' not in result_dict_cover:
        result_dict_cover['HMMER'] = []
    fpr, tpr, auc = hmmer(query, 'phmmer', 1, 'HMMER', roc_top, result_dict_cover)
    result_dict['HMMER'][query] = auc
    ax.plot(fpr, tpr, label=f'HMMER AUC={auc:.4f}')

    # Proposed
    if 'proposed' not in result_dict_cover:
        result_dict_cover['proposed'] = []
    try:
        fpr, tpr, auc = proposed(query, gap_open, gap_extend, roc_top, result_dict_cover, fold, clst, blosum_flg, kei,
                                 lam)
    except Exception as e:
        logging.exception(e)
        return
    if 'proposed' not in result_dict:
        result_dict['proposed'] = {}
    result_dict['proposed'][query] = auc
    ax.plot(fpr, tpr, label=f'Proposed AUC={auc:.4f}')

    # Proposed 2
    if 'proposed2' not in result_dict_cover:
        result_dict_cover['proposed2'] = []
    try:
        fpr, tpr, auc = proposed2(query, gap_open, gap_extend, roc_top, result_dict_cover, blast_ranking, fold, clst,
                                  blosum_flg, kei, lam)
    except Exception as e:
        logging.exception(e)
        return
    if 'proposed2' not in result_dict:
        result_dict['proposed2'] = {}
    result_dict['proposed2'][query] = auc
    ax.plot(fpr, tpr, label=f'Proposed2 AUC={auc:.4f}')
    # Summarize
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)
    ax.set_xlabel('# of false positives')
    ax.set_ylabel('# of true positives')
    ax.set_title(f'ROC{roc_top} [query={query}, Open={gap_open}, Extend={gap_extend}]')
    rdir = Path(f'results/roc{roc_top}/op{gap_open}_ex{gap_extend}_{blosum_flg}')
    rdir.mkdir(exist_ok=True, parents=True)
    fig.savefig(rdir / f'{query}.png')
    matplotlib.pyplot.close()


def summarize(result_dict, result_dict_cover, roc_top, gap_open, gap_extend, blosum_flg, kei, lam):
    result_df = pandas.DataFrame(result_dict)
    # result_df_cover = pandas.DataFrame(result_dict_cover)
    # fig: matplotlib.figure.Figure = None
    # ax: matplotlib.axes.Axes = None
    fig, ax = matplotlib.pyplot.subplots(figsize=(20, 10))
    ax.set_title(f'roc{roc_top}_open{gap_open}_extend{gap_extend}_blosum62{blosum_flg}_K{kei}_lambda{lam}.png')
    sns.boxplot(data=result_df, ax=ax, showmeans=True)
    fig.savefig(f'boxplot_roc{roc_top}_open{gap_open}_extend{gap_extend}_blosum62{blosum_flg}_K{kei}_lambda{lam}.png')
    matplotlib.pyplot.close()
    fig, ax = matplotlib.pyplot.subplots(figsize=(20, 10))
    ax.boxplot(x=[result_dict_cover['DELTA-BLAST_iter1'],
                  result_dict_cover['PSI-BLAST_iter1'],
                  result_dict_cover['PSI-BLAST_PSSM_iter1'],
                  result_dict_cover['PSI-BLAST_iter3'],
                  result_dict_cover['PSI-BLAST_iter5'],
                  result_dict_cover['HMMER'],
                  result_dict_cover['proposed'],
                  result_dict_cover['proposed2']],
               labels=['DELTA-BLAST_iter1', 'PSI-BLAST_iter1', 'PSI-BLAST_PSSM_iter1',
                       'PSI-BLAST_iter3', 'PSI-BLAST_iter5', 'HMMER', 'proposed', 'proposed2'],
               showmeans=True)
    fig.savefig(f'cover_roc{roc_top}_open{gap_open}_extend{gap_extend}_blosum62{blosum_flg}_K{kei}_lambda{lam}.png')
    matplotlib.pyplot.close()


def grid_search(fold, clst, gap_open, gap_extend, roc_top, blosum_flg, lamb, k, grid_dict):
    result_dict = {}
    result_dict_cover = {}
    blast_ranking = {}

    for t in tqdm(test_data):
        make_result(t[0], gap_open, gap_extend, blosum_flg, roc_top, result_dict, result_dict_cover, blast_ranking,
                    fold, clst, k, lamb)

    summarize(result_dict, result_dict_cover, roc_top, gap_open, gap_extend, blosum_flg, k, lamb)

    result_df = pandas.DataFrame(result_dict)
    grid_name = f'roc{roc_top}_b{blosum_flg}_l{lamb}_K{k}'
    grid_dict['DELTA-BLAST_iter1'][grid_name] = result_df['DELTA-BLAST_iter1'].mean()
    grid_dict['PSI-BLAST_iter1'][grid_name] = result_df['PSI-BLAST_iter1'].mean()
    grid_dict['PSI-BLAST_PSSM_iter1'][grid_name] = result_df['PSI-BLAST_PSSM_iter1'].mean()
    grid_dict['PSI-BLAST_iter3'][grid_name] = result_df['PSI-BLAST_iter3'].mean()
    grid_dict['PSI-BLAST_iter5'][grid_name] = result_df['PSI-BLAST_iter5'].mean()
    grid_dict['HMMER'][grid_name] = result_df['HMMER'].mean()
    grid_dict['proposed'][grid_name] = result_df['proposed'].mean()
    grid_dict['proposed2'][grid_name] = result_df['proposed2'].mean()


def iter_grid_search():
    grid_dict = {
        'DELTA-BLAST_iter1': {},
        'PSI-BLAST_iter1': {},
        'PSI-BLAST_PSSM_iter1': {},
        'PSI-BLAST_iter3': {},
        'PSI-BLAST_iter5': {},
        'HMMER': {},
        'proposed': {},
        'proposed2': {}
    }
    for roc in (100,):
        for lamb in (0.001, 0.005, 0.01, 0.05, 0.1):
            for K in (0.1, 0.5, 1, 3, 5, 10):
                grid_search(100, 100, 25, 9, roc, 'on', lamb, K, grid_dict)
    # grid_search(100, 100, 25, 18, 100, 'on', 0.001, 5, grid_dict)
    grid_df = pandas.DataFrame(grid_dict)
    grid_df['pro2_DEL1'] = grid_df['proposed2'] - grid_df['DELTA-BLAST_iter1']
    grid_df['pro_PSI3'] = grid_df['proposed'] - grid_df['PSI-BLAST_iter3']
    grid_df['pro_HMMR'] = grid_df['proposed'] - grid_df['HMMER']
    grid_df['pro_PSS1'] = grid_df['proposed'] - grid_df['PSI-BLAST_PSSM_iter1']
    grid_df.to_csv('grid_search.csv')


iter_grid_search()
