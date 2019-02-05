from pathlib import Path
import os
import subprocess
import random
import re

from Bio import SeqIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_protein
from Bio import SearchIO
from Bio.SubsMat import MatrixInfo
from Bio import pairwise2
from Bio.Blast.Applications import NcbipsiblastCommandline, NcbideltablastCommandline
import numpy as np
from tqdm import tqdm


# ToDo: Support PDB chains
def replace_missing_residues(template_alignment, pdb):
    template_pdb = str(SeqIO.read(pdb, 'pdb-atom').seq).replace('X', '')
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    alignment = next(aligner.align(template_alignment.replace('-', ''), template_pdb))
    a, _, b = str(alignment).splitlines()
    a = list(a)
    b = list(b)
    assert len(a) == len(b)
    for i, _ in enumerate(a):
        if b[i] == '-':
            a[i] = '@'
    a = ''.join(a).replace('-', '')
    a = list(a)
    for i, res in enumerate(template_alignment):
        if res == '-':
            a.insert(i, '-')
    a = ''.join(a).replace('@', '-')
    assert len(a) == len(template_alignment)
    return a


def _pp(path):
    seq = []
    for line in Path(path).read_text().splitlines():
        token = line.rstrip('\r\n').split()
        if len(token) == 0:
            continue
        if re.match(r'\d+', token[0]):
            seq.append(token[1])
    return Seq(''.join(seq), generic_protein)


class MachinaModel:
    @staticmethod
    def _get_top_aln(search_result, n_top):
        result_d = {}
        for r in np.load(search_result):
            key = list(r.keys())[0]
            bst = np.argmax([x[2] for x in r[key]])
            result_d[key] = r[key][bst]
        return sorted(result_d.items(), key=lambda x: x[1][2], reverse=True)[:n_top]

    @classmethod
    def generate_protein_models_from_search(cls, search_result, query, out_dir, n_top=10):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for r in cls._get_top_aln(search_result, n_top):
            template = r[0]
            if Path(f'{out_dir}/{template}.pdb').exists():
                continue
            if not Path(f'data/scop_e/{template[2:4]}/{template}.ent').exists():
                continue
            aln = r[1]
            pir_file = f'{out_dir}/{template}.pir'
            tseq = replace_missing_residues(aln[1], f'data/scop_e/{template[2:4]}/{template}.ent')
            SeqIO.write([
                SeqRecord(Seq(aln[0], generic_protein), id=query, name='',
                          description=f'sequence:{query}::::::::'),
                SeqRecord(Seq(tseq, generic_protein), id=template, name='',
                          description=f'structureX:{template}::{template[5].upper()}::{template[5].upper()}::::')
            ], pir_file, 'pir')
            arg = ['/opt/modeller-9.20/bin/modpy.sh', 'python3', 'machina/modeller_script.py',
                   pir_file, template, query, f'data/scop_e/{template[2:4]}']
            subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    @staticmethod
    def generate_protein_model(query: str, template: str, alignments_list: str, out_dir: str, template_dir: str):
        aln = np.load(alignments_list)
        best = aln[np.argmax([_[2] for _ in aln])]
        pir_file = f'{out_dir}/{template}.pir'
        tseq = replace_missing_residues(best[1], f'{template_dir}/{template}.ent')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        SeqIO.write([
            SeqRecord(Seq(best[0], generic_protein), id=query, name='', description=f'sequence:{query}::::::::'),
            SeqRecord(Seq(tseq, generic_protein), id=template, name='',
                      description=f'structureX:{template}::{template[5].upper()}::{template[5].upper()}::::')
        ], pir_file, 'pir')
        arg = ['/opt/modeller-9.20/bin/modpy.sh', 'python3', Path(__file__).parent.resolve()/'modeller_script.py',
               pir_file, template, query, template_dir]
        subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


class BLASTModel:
    def __init__(self, algo, blast_db_dir):
        self.algo = algo
        self.blast_db_dir = blast_db_dir

    def generate_search_result(self, blast_dir, blast_db):
        for query_sid in tqdm(test_data):
            Path(f'{blast_dir}/{query_sid}').mkdir(parents=True, exists=True)
            xml_f = Path(f'{blast_dir}/{query_sid}/{query_sid}.xml')
            pssm_f = Path(f'data/test/{query_sid}.pssm')
            if self.algo == 'psiblast_iter3':
                NcbipsiblastCommandline(in_pssm=pssm_f.as_posix(),
                                        db=f'{blast_db}/astral-scopdom-seqres-gd-sel-gs-bib-40-1.75_train_only',
                                        num_threads=int(os.cpu_count()),
                                        num_iterations=3,
                                        max_target_seqs=500,
                                        evalue=999999,
                                        outfmt=5,
                                        out=xml_f.as_posix())()

    def _get_best_hsp(self, blast_result, template_sid):
        hits = [x.id for x in blast_result.hits]
        if template_sid not in hits:
            return None
        hit = blast_result.hits[hits.index(template_sid)]
        return hit.hsps[0].aln

    def generate_pairwise_alignment(self, query_id: str, target_id: str, out_dir: str, pssm_dir: str):
        Path(f'{out_dir}/{query_id}').mkdir(parents=True, exist_ok=True)
        SeqIO.write(
            SeqRecord(_pp(f'{pssm_dir}/{query_id[2:4]}/{query_id}.mtx'), id=query_id), 'query.fasta', 'fasta')
        SeqIO.write(
            SeqRecord(_pp(f'{pssm_dir}/{target_id[2:4]}/{target_id}.mtx'), id=target_id), 'subject.fasta', 'fasta')
        if self.algo == 'psiblast':
            if not Path(f'{pssm_dir}/{query_id[2:4]}/{query_id}.pssm').exists():
                NcbipsiblastCommandline(db=f'{self.blast_db_dir}/uniref90', num_iterations=3,
                                        out_pssm=f'{pssm_dir}/{query_id[2:4]}/{query_id}.pssm', query='query.fasta',
                                        save_pssm_after_last_round=True, num_threads=os.cpu_count())()
            NcbipsiblastCommandline(in_pssm=f'{pssm_dir}/{query_id[2:4]}/{query_id}.pssm',
                                    subject='subject.fasta', outfmt=5, out=f'{out_dir}/{query_id}/{target_id}.xml')()
        elif self.algo == 'deltablast':
            NcbideltablastCommandline(subject='subject.fasta', rpsdb=f'{self.blast_db_dir}/cdd_delta',
                                      outfmt=5, out=f'{out_dir}/{query_id}/{target_id}.xml', query='query.fasta')()

    def generate_models_from_search(self, blast_dir):
        top = 10
        for query_sid in tqdm(test_data):
            blast_result = next(SearchIO.parse(f'{blast_dir}/{query_sid}.xml', 'blast-xml'))
            Path(f'{blast_dir}_top{top}/{query_sid}').mkdir(parents=True, exist_ok=True)
            for hit in tqdm(blast_result.hits[:top]):
                hsp = hit.hsps[0]
                template_id = hsp.aln[1].id
                if Path(f'{blast_dir}_top{top}/{query_sid}/{template_id}.pdb').exists():
                    continue
                if not Path(f'data/scop_e/{template_id[2:4]}/{template_id}.ent').exists():
                    continue
                tseq = replace_missing_residues(str(hsp.aln[1].seq), f'data/scop_e/{template_id[2:4]}/{template_id}.ent')
                chain = template_id[5].upper()
                pir_file = f'{blast_dir}_top{top}/{query_sid}/{template_id}.pir'
                SeqIO.write([
                    SeqRecord(Seq(str(hsp.aln[0].seq), generic_protein), id=query_sid, name='',
                              description=f'sequence:{query_sid}::::::::'),
                    SeqRecord(Seq(tseq, generic_protein), id=template_id, name='',
                              description=f'structureX:{template_id}::{chain}::{chain}::::')
                ], pir_file, 'pir')
                arg = ['modpy.sh', 'python3', 'modeller_script.py', pir_file, template_id, query_sid, f'data/scop_e/{template_id[2:4]}']
                subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    def generate_models(self, blast_dir):
        for sid in tqdm(test_data):
            sunid = scop_root.getDomainBySid(sid).getAscendent('sf').sunid
            domains = scop_root.getNodeBySunid(sunid).getDescendents('px')
            for d in tqdm(domains):
                try:
                    blast_result = next(SearchIO.parse(f'{blast_dir}/{sid}/{d.sid}.xml', 'blast-xml'))
                except FileNotFoundError:
                    continue
                best = self._get_best_hsp(blast_result, d.sid)
                if best is None:
                    continue
                if Path(f'{blast_dir}/{sid}/{d.sid}.pdb').exists():
                    continue
                if not Path(f'data/scop_e/{d.sid[2:4]}/{d.sid}.ent').exists():
                    continue
                pir_file = f'{blast_dir}/{sid}/{d.sid}.pir'
                tseq = replace_missing_residues(str(best[1].seq), f'data/scop_e/{d.sid[2:4]}/{d.sid}.ent')
                SeqIO.write([
                    SeqRecord(Seq(str(best[0].seq), generic_protein), id=sid, name='',
                              description=f'sequence:{sid}::::::::'),
                    SeqRecord(Seq(tseq, generic_protein), id=d.sid, name='',
                              description=f'structureX:{d.sid}::{d.sid[5].upper()}::{d.sid[5].upper()}::::')
                ], pir_file, 'pir')
                arg = ['modpy.sh', 'python3', 'modeller_script.py', pir_file, d.sid, sid, f'data/scop_e/{d.sid[2:4]}']
                subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    def generate_protein_model(self, query, template, alignments_list, out_dir):
        if Path(f'{out_dir}/{template}.pdb').exists():
            return
        if not Path(f'data/scop_e/{template[2:4]}/{template}.ent').exists():
            return
        blast_result = next(SearchIO.parse(f'{out_dir}/{template}.xml', 'blast-xml'))
        best = self._get_best_hsp(blast_result, template)
        if best is None:
            return
        tseq = replace_missing_residues(str(best[1].seq), f'data/scop_e/{template[2:4]}/{template}.ent')
        pir_file = f'{out_dir}/{template}.pir'
        SeqIO.write([
            SeqRecord(Seq(str(best[0].seq), generic_protein), id=query, name='',
                      description=f'sequence:{query}::::::::'),
            SeqRecord(Seq(tseq, generic_protein), id=template, name='',
                      description=f'structureX:{template}::{template[5].upper()}::{template[5].upper()}::::')
        ], pir_file, 'pir')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # arg = ['mod9.20', 'machina/modeller_script.py', pir_file, template, query, f'data/scop_e/{d.sid[2:4]}']
        arg = ['/opt/modeller-9.20/bin/modpy.sh', 'python3', 'machina/modeller_script.py',
               pir_file, template, query, f'data/scop_e/{template[2:4]}']
        subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


class TMalignModel:
    def generate_models(self, out_dir):
        aln_db = SeqIO.index('data/scop40_structural_alignment.fasta', 'fasta')
        for sid in tqdm(test_data):
            Path(f'{out_dir}/{sid}').mkdir(parents=True, exist_ok=True)
            sunid = scop_root.getDomainBySid(sid).getAscendent('sf').sunid
            domains = scop_root.getNodeBySunid(sunid).getDescendents('px')
            for d in tqdm(domains):
                pir_file = f'{out_dir}/{sid}/{d.sid}.pir'
                try:
                    SeqIO.write([
                        SeqRecord(Seq(str(aln_db[f'{sid}&{d.sid}'].seq), generic_protein), id=sid, name='',
                                  description=f'sequence:{sid}::::::::'),
                        SeqRecord(Seq(str(aln_db[f'{d.sid}&{sid}'].seq), generic_protein), id=d.sid, name='',
                                  description=f'structureX:{d.sid}::{d.sid[5].upper()}::{d.sid[5].upper()}::::')
                    ], pir_file, 'pir')
                except KeyError:
                    continue
                arg = ['mod9.20', 'modeller_script.py', pir_file, d.sid, sid, f'data/scop_e/{d.sid[2:4]}']
                subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    def generate_protein_model(self, query, template, alignments_list, out_dir):
        if Path(f'{out_dir}/{template}.pdb').exists():
            return
        if not Path(f'data/scop_e/{template[2:4]}/{template}.ent').exists():
            return
        pir_file = f'{out_dir}/{template}.pir'
        tseq = replace_missing_residues(str(alignments_list[f'{template}&{query}'].seq),
                                        f'data/scop_e/{template[2:4]}/{template}.ent')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        SeqIO.write([
            SeqRecord(Seq(str(alignments_list[f'{query}&{template}'].seq).replace('\n', ''), generic_protein),
                      id=query, name='', description=f'sequence:{query}::::::::'),
            SeqRecord(Seq(tseq, generic_protein), id=template, name='',
                      description=f'structureX:{template}::{template[5].upper()}::{template[5].upper()}::::')
        ], pir_file, 'pir')
        # arg = ['mod9.20', 'machina/modeller_script.py', pir_file, template, query, f'data/scop_e/{d.sid[2:4]}']
        arg = ['/opt/modeller-9.20/bin/modpy.sh', 'python3', 'machina/modeller_script.py',
               pir_file, template, query, f'data/scop_e/{template[2:4]}']
        subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


class SWModel:
    def __init__(self, gap_open, gap_extend):
        self.gap_open = gap_open
        self.gap_extend = gap_extend

    def generate_models(self):
        aln_db = SeqIO.index('data/train/scop40_structural_alignment.fasta', 'fasta')
        out_dir = f'data/.sw_aln/open{-self.gap_open}_extend{-self.gap_extend}'
        for sid in tqdm(test_data):
            Path(f'{out_dir}/{sid}').mkdir(parents=True, exist_ok=True)
            sunid = scop_root.getDomainBySid(sid).getAscendent('sf').sunid
            domains = scop_root.getNodeBySunid(sunid).getDescendents('px')
            for d in tqdm(domains):
                try:
                    ali = pairwise2.align.localds(str(aln_db[f'{sid}&{d.sid}'].seq.ungap('-')),
                                                  str(aln_db[f'{d.sid}&{sid}'].seq.ungap('-')),
                                                  MatrixInfo.blosum62, self.gap_open, self.gap_extend)
                except KeyError:
                    continue
                pir_file = f'{out_dir}/{sid}/{d.sid}.pir'
                best = ali[np.argmax([x[2] for x in ali])]
                SeqIO.write([
                    SeqRecord(Seq(str(best[0]), generic_protein), id=sid, name='',
                              description=f'sequence:{sid}::::::::'),
                    SeqRecord(Seq(str(best[1]), generic_protein), id=d.sid, name='',
                              description=f'structureX:{d.sid}::{d.sid[5].upper()}::{d.sid[5].upper()}::::')
                ], pir_file, 'pir')
                arg = ['mod9.20', 'modeller_script.py', pir_file, d.sid, sid, f'data/scop_e/{d.sid[2:4]}']
                arg = ['/usr/lib/modeller9.20/bin/modpy.sh', 'python3', 'modeller_script.py',
                       pir_file, d.sid, sid, f'data/scop_e/{d.sid[2:4]}']
                subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    def generate_protein_model(self, query, template, alignments_list, out_dir):
        if Path(f'{out_dir}/{template}.pdb').exists():
            return
        if not Path(f'data/scop_e/{template[2:4]}/{template}.ent').exists():
            return
        ali = pairwise2.align.localds(str(alignments_list[f'{query}&{template}'].seq.ungap('-')),
                                      str(alignments_list[f'{template}&{query}'].seq.ungap('-')),
                                      MatrixInfo.blosum62, self.gap_open, self.gap_extend)
        best = ali[np.argmax([_[2] for _ in ali])]
        tseq = replace_missing_residues(str(best[1]), f'data/scop_e/{template[2:4]}/{template}.ent')
        pir_file = f'{out_dir}/{template}.pir'
        SeqIO.write([
            SeqRecord(Seq(str(best[0]), generic_protein), id=query, name='',
                      description=f'sequence:{query}::::::::'),
            SeqRecord(Seq(tseq, generic_protein), id=template, name='',
                      description=f'structureX:{template}::{template[5].upper()}::{template[5].upper()}::::')
        ], pir_file, 'pir')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # arg = ['mod9.20', 'machina/modeller_script.py', pir_file, template, query, f'data/scop_e/{d.sid[2:4]}']
        arg = ['/opt/modeller-9.20/bin/modpy.sh', 'python3', 'machina/modeller_script.py',
               pir_file, template, query, f'data/scop_e/{template[2:4]}']
        subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


class HHSearchModel:
    def generate_query_a3m(self, hhsearch_dir, db_path='/DB/uniclust30_2017_10'):
        aln_db = SeqIO.index('data/scop40_structural_alignment.fasta', 'fasta')
        aln = {}
        for i in tqdm(aln_db):
            domkey = i.split('&')[0]
            aln[domkey] = SeqRecord(aln_db[i].seq.ungap('-'), id=domkey, name='', description='')
        tlist = list(test_data.keys())
        random.shuffle(tlist)
        for query_sid in tqdm(tlist):
            Path(f'{hhsearch_dir}/{query_sid}').mkdir(parents=True, exist_ok=True)
            if Path(f'{hhsearch_dir}/{query_sid}/{query_sid}.a3m').exists():
                continue
            SeqIO.write(aln[query_sid], f'{hhsearch_dir}/{query_sid}/{query_sid}.fasta', 'fasta')
            arg = ['hhsearch', '-i', f'{query_sid}.fasta', '-d', db_path,
                   '-o', '/dev/null', '-oa3m', f'{query_sid}.a3m', '-cpu', '64']
            subprocess.run(arg, cwd=f'{hhsearch_dir}/{query_sid}',
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def generate_search_result(self, hhsearch_dir, db_dir='/DB/scop40'):
        for query_sid in tqdm(test_data):
            arg = ['hhsearch', '-i', f'{query_sid}.a3m', '-d', db_dir, '-cpu', '64', '-z', '500', '-b', '500']
            subprocess.run(arg, cwd=f'{hhsearch_dir}/{query_sid}',
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def generate_search_result_global(self, hhsearch_dir, db_dir='/DB/scop40'):
        for query_sid in tqdm(test_data):
            arg = ['hhsearch', '-i', f'{query_sid}.a3m', '-d', db_dir, '-cpu', '64', '-z', '500', '-b', '500',
                   '-glob', '-o', f'{query_sid}_global_aln.hhr']
            subprocess.run(arg, cwd=f'{hhsearch_dir}/{query_sid}',
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def generate_pairwise_aln(self, hhsearch_dir, db_dir):
        hhm = f'{db_dir}/scop40_hhm.ffdata'
        hie_d = {}
        for line in [_ for _ in Path(hhm).read_text().splitlines() if _.startswith('NAME ')]:
            domain_sid = line.split()[1]
            sf_sccs = '.'.join(line.split()[2].split('.')[:3])
            if sf_sccs not in hie_d:
                hie_d[sf_sccs] = [domain_sid]
            else:
                hie_d[sf_sccs].append(domain_sid)
        names = {l.split()[0]: (int(l.split()[1]), int(l.split()[2]))
                 for l in Path(f'{db_dir}/scop40_a3m.ffindex').read_text().splitlines()}
        with Path(f'{db_dir}/scop40_a3m.ffdata').open() as f:
            for query_sid in tqdm(test_data):
                query_sf_sccs = test_data[query_sid][1]
                if query_sf_sccs not in hie_d:
                    print(f'!!! {query_sf_sccs} is not in HHsearch SCOP40 DB !!!')
                    continue
                for tmpl_sid in tqdm([_ for _ in hie_d[query_sf_sccs] if _ != query_sid]):
                    f.seek(names[tmpl_sid][0])
                    a3m = f.read(names[tmpl_sid][1])
                    Path(f'{hhsearch_dir}/{query_sid}/{tmpl_sid}.a3m').write_text(a3m)
                    arg = ['hhalign', '-i', f'{query_sid}.a3m', '-t', f'{tmpl_sid}.a3m', '-o', f'{tmpl_sid}.hhr', '-glob']
                    subprocess.run(arg, cwd=f'{hhsearch_dir}/{query_sid}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    raw_lines = Path(f'{hhsearch_dir}/{query_sid}/{tmpl_sid}.hhr').read_text().splitlines()
                    result_points = [i for i, l in enumerate(raw_lines) if l.startswith('No ')]
                    assert len(result_points) == 1
                    lines = raw_lines[result_points[0]:]
                    aln_lines = [l for l in lines if l.startswith('Q ') and 'Consensus' not in l]
                    if len(aln_lines) == 0:
                        continue
                    qseq = Seq(''.join([l.split()[3] for l in aln_lines]), generic_protein)
                    tmpl_sid = lines[1].split()[0][1:]
                    aln_lines = [l for l in lines if l.startswith('T ') and 'Consensus' not in l]
                    tseq = ''.join([l.split()[3] for l in aln_lines])
                    if Path(f'data/scop_e/{tmpl_sid[2:4]}/{tmpl_sid}.ent').exists():
                        tseq = replace_missing_residues(tseq, f'data/scop_e/{tmpl_sid[2:4]}/{tmpl_sid}.ent')
                    elif Path(f'data/scop_e_all/{tmpl_sid[2:4]}/{tmpl_sid}.ent').exists():
                        tseq = replace_missing_residues(tseq, f'data/scop_e_all/{tmpl_sid[2:4]}/{tmpl_sid}.ent')
                    else:
                        print(f'{tmpl_sid}.ent is missing')
                        continue
                    tseq = Seq(tseq, generic_protein)
                    pir_file = f'{hhsearch_dir}/{query_sid}/{tmpl_sid}.pir'
                    SeqIO.write([
                        SeqRecord(qseq, id=query_sid, name='', description=f'sequence:{query_sid}::::::::'),
                        SeqRecord(tseq, id=tmpl_sid, name='',
                                  description=f'structureX:{tmpl_sid}::{tmpl_sid[5].upper()}::{tmpl_sid[5].upper()}::::')
                    ], pir_file, 'pir')
                    Path(f'{hhsearch_dir}/{query_sid}/{tmpl_sid}.hhr').unlink()

    def generate_models_from_search(self, hhsearch_dir):
        top = 10
        for query_sid in tqdm(test_data):
            raw_lines = Path(f'{hhsearch_dir}/{query_sid}/{query_sid}.hhr').read_text().splitlines()
            result_points = [i for i, l in enumerate(raw_lines) if l.startswith('No ')]
            assert len(result_points) > 5
            for i, _ in enumerate(result_points[:top]):
                if i == len(result_points)-1:
                    lines = raw_lines[result_points[i]:]
                else:
                    lines = raw_lines[result_points[i]:result_points[i+1]]
                aln_lines = [l for l in lines if l.startswith('Q ') and 'Consensus' not in l]
                qseq = Seq(''.join([l.split()[3] for l in aln_lines]), generic_protein)
                tmpl_sid = lines[1].split()[0][1:]
                aln_lines = [l for l in lines if l.startswith('T ') and 'Consensus' not in l]
                tseq = Seq(''.join([l.split()[3] for l in aln_lines]), generic_protein)
                Path(f'{hhsearch_dir}/{query_sid}/top{top}').mkdir(parents=True, exist_ok=True)
                pir_file = f'{hhsearch_dir}/{query_sid}/top{top}/{tmpl_sid}.pir'
                SeqIO.write([
                    SeqRecord(qseq, id=query_sid, name='', description=f'sequence:{query_sid}::::::::'),
                    SeqRecord(tseq, id=tmpl_sid, name='',
                              description=f'structureX:{tmpl_sid}::{tmpl_sid[5].upper()}::{tmpl_sid[5].upper()}::::')
                ], pir_file, 'pir')
                # arg = ['mod9.18', 'modeller_script.py', pir_file, tmpl_sid, query_sid, f'data/scop_e/{tmpl_sid[2:4]}']
                arg = ['modpy.sh', 'python3', 'modeller_script.py', pir_file, tmpl_sid, query_sid, f'data/scop_e/{tmpl_sid[2:4]}']
                subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    def generate_models_from_search_global(self, hhsearch_dir):
        top = 10
        for query_sid in tqdm(test_data):
            raw_lines = Path(f'{hhsearch_dir}/{query_sid}/{query_sid}_global_aln.hhr').read_text().splitlines()
            result_points = [i for i, l in enumerate(raw_lines) if l.startswith('No ')]
            assert len(result_points) > 5
            for i, _ in enumerate(result_points[:top]):
                if i == len(result_points)-1:
                    lines = raw_lines[result_points[i]:]
                else:
                    lines = raw_lines[result_points[i]:result_points[i+1]]
                aln_lines = [l for l in lines if l.startswith('Q ') and 'Consensus' not in l]
                qseq = Seq(''.join([l.split()[3] for l in aln_lines]), generic_protein)
                tmpl_sid = lines[1].split()[0][1:]
                aln_lines = [l for l in lines if l.startswith('T ') and 'Consensus' not in l]
                tseq = ''.join([l.split()[3] for l in aln_lines])
                scop_e_dir = 'data/scop_e_all'
                if Path(f'{hhsearch_dir}/{query_sid}/top{top}_global_aln/{tmpl_sid}.pdb').exists():
                    continue
                if not Path(f'{scop_e_dir}/{tmpl_sid[2:4]}/{tmpl_sid}.ent').exists():
                    continue
                tseq = replace_missing_residues(tseq, f'{scop_e_dir}/{tmpl_sid[2:4]}/{tmpl_sid}.ent')
                tseq = Seq(tseq, generic_protein)
                Path(f'{hhsearch_dir}/{query_sid}/top{top}_global_aln').mkdir(parents=True, exist_ok=True)
                pir_file = f'{hhsearch_dir}/{query_sid}/top{top}_global_aln/{tmpl_sid}.pir'
                SeqIO.write([
                    SeqRecord(qseq, id=query_sid, name='', description=f'sequence:{query_sid}::::::::'),
                    SeqRecord(tseq, id=tmpl_sid, name='',
                              description=f'structureX:{tmpl_sid}::{tmpl_sid[5].upper()}::{tmpl_sid[5].upper()}::::')
                ], pir_file, 'pir')
                # arg = ['mod9.18', 'modeller_script.py', pir_file, tmpl_sid, query_sid, f'data/scop_e/{tmpl_sid[2:4]}']
                arg = ['modpy.sh', 'python3', 'modeller_script.py', pir_file, tmpl_sid, query_sid, f'{scop_e_dir}/{tmpl_sid[2:4]}']
                # subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                subprocess.run(arg, universal_newlines=True)

    def generate_models(self, hhsearch_dir, db_dir):
        hhm = f'{db_dir}/scop40_hhm.ffdata'
        hie_d = {}
        for line in [_ for _ in Path(hhm).read_text().splitlines() if _.startswith('NAME ')]:
            domain_sid = line.split()[1]
            sf_sccs = '.'.join(line.split()[2].split('.')[:3])
            if sf_sccs not in hie_d:
                hie_d[sf_sccs] = [domain_sid]
            else:
                hie_d[sf_sccs].append(domain_sid)
        for query_sid in tqdm(test_data):
            query_sf_sccs = test_data[query_sid][1]
            if query_sf_sccs not in hie_d:
                print(f'!!! {query_sf_sccs} is not in HHsearch SCOP40 DB !!!')
                continue
            for tmpl_sid in tqdm([_ for _ in hie_d[query_sf_sccs] if _ != query_sid]):
                if Path(f'{hhsearch_dir}/{query_sid}/{tmpl_sid}.pdb').exists():
                    continue
                if Path(f'data/scop_e/{tmpl_sid[2:4]}/{tmpl_sid}.ent').exists():
                    atom_dir = f'data/scop_e/{tmpl_sid[2:4]}'
                elif Path(f'data/scop_e_all/{tmpl_sid[2:4]}/{tmpl_sid}.ent').exists():
                    atom_dir = f'data/scop_e_all/{tmpl_sid[2:4]}'
                else:
                    continue
                arg = ['modpy.sh', 'python3', 'modeller_script.py',
                       f'{hhsearch_dir}/{query_sid}/{tmpl_sid}.pir', tmpl_sid, query_sid, atom_dir]
                # subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                subprocess.run(arg, universal_newlines=True)

    def generate_protein_model(self, query, template, alignments_list, out_dir):
        if Path(f'{out_dir}/{template}.pdb').exists():
            return
        if Path(f'data/scop_e/{template[2:4]}/{template}.ent').exists():
            atom_dir = f'data/scop_e/{template[2:4]}'
        elif Path(f'data/scop_e_all/{template[2:4]}/{template}.ent').exists():
            atom_dir = f'data/scop_e_all/{template[2:4]}'
        else:
            return
        # tseq = replace_missing_residues(str(best[1]), f'data/scop_e/{template[2:4]}/{template}.ent')
        pir_file = f'{out_dir}/{template}.pir'
        # SeqIO.write([
        #     SeqRecord(Seq(str(best[0]), generic_protein), id=query, name='',
        #               description=f'sequence:{query}::::::::'),
        #     SeqRecord(Seq(tseq, generic_protein), id=template, name='',
        #               description=f'structureX:{template}::{template[5].upper()}::{template[5].upper()}::::')
        # ], pir_file, 'pir')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # arg = ['mod9.20', 'machina/modeller_script.py', pir_file, template, query, f'data/scop_e/{d.sid[2:4]}']
        arg = ['/opt/modeller-9.20/bin/modpy.sh', 'python3', 'machina/modeller_script.py',
               pir_file, template, query, atom_dir]
        subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
