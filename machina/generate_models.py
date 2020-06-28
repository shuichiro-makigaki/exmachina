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


def replace_missing_residues(template_alignment, template_id, chain, pdb):
    template_pdb = [str(_.seq) for _ in SeqIO.parse(pdb, 'pdb-atom') if _.id == f'{template_id}:{chain}']
    try:
        template_pdb = template_pdb[0].replace('X', '')
    except IndexError:
        print(template_id)
        print(chain)

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
    def __init__(self, mod_bin):
        self.mod_bin = mod_bin

    def generate_protein_model(self, query_id: str, template_id: str, chain: str,
                               alignments_list: Path, template_file: Path, out_dir: Path):
        aln = np.load(alignments_list)
        best = aln[np.argmax([float(_[2]) for _ in aln])]
        pir_file = out_dir/'alignment.pir'
        tseq = replace_missing_residues(best[1], template_id, chain, template_file.as_posix())
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        SeqIO.write([
            SeqRecord(Seq(best[0], generic_protein), id=query_id, name='',
                      description=f'sequence:{query_id}::::::::'),
            SeqRecord(
                Seq(tseq, generic_protein), id=template_id, name='',
                description=f'structureX:{template_id}::{chain}::{chain}::::')
        ], pir_file, 'pir')
        arg = [self.mod_bin, Path(__file__).parent.resolve()/'modeller_script.py',
               pir_file.resolve(), query_id, template_id, template_file.parent.resolve()]
        res = subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=out_dir)
        print(res.stdout)
        print(res.stderr)


class BLASTModel:
    def __init__(self, algo, blast_db_dir, modpy_sh_path='/opt/modeller-9.20/bin/modpy.sh'):
        self.algo = algo
        self.blast_db_dir = blast_db_dir
        self.modpysh = modpy_sh_path

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
            NcbipsiblastCommandline(in_pssm=f'{pssm_dir}/{query_id[2:4]}/{query_id}.pssm', evalue=99999,
                                    subject='subject.fasta', outfmt=5, out=f'{out_dir}/{query_id}/{target_id}.xml')()
        elif self.algo == 'deltablast':
            NcbideltablastCommandline(subject='subject.fasta', rpsdb=f'{self.blast_db_dir}/cdd_delta', evalue=99999,
                                      outfmt=5, out=f'{out_dir}/{query_id}/{target_id}.xml', query='query.fasta')()
        Path('query.fasta').unlink()
        Path('subject.fasta').unlink()

    def generate_protein_model(self, query: str, template: str, blast_xml_path: str, out_dir: str, template_dir: str):
        hits = [_ for _ in SearchIO.read(blast_xml_path, 'blast-xml').hits if _.id == template]
        assert len(hits) == 1
        best = hits[0].hsps[0].aln
        tseq = replace_missing_residues(str(best[1].seq), f'{template_dir}/{template}.ent')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pir_file = f'{out_dir}/{template}.pir'
        SeqIO.write([
            SeqRecord(Seq(str(best[0].seq), generic_protein), id=query, name='',
                      description=f'sequence:{query}::::::::'),
            SeqRecord(Seq(tseq, generic_protein), id=template, name='',
                      description=f'structureX:{template}::{template[5].upper()}::{template[5].upper()}::::')
        ], pir_file, 'pir')
        arg = [self.modpysh, 'python3', Path(__file__).parent.resolve()/'modeller_script.py',
               pir_file, template, query, template_dir]
        subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


class TMalignModel:
    def __init__(self, modpy_sh_path='/opt/modeller-9.20/bin/modpy.sh'):
        self.modpysh = modpy_sh_path

    def generate_protein_model(self, query: str, template: str, alignments: SeqIO, template_dir: str, out_dir: str):
        pir_file = f'{out_dir}/{template}.pir'
        tseq = replace_missing_residues(str(alignments[f'{template}&{query}'].seq),
                                        f'{template_dir}/{template}.ent')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        SeqIO.write([
            SeqRecord(Seq(str(alignments[f'{query}&{template}'].seq).replace('\n', ''), generic_protein),
                      id=query, name='', description=f'sequence:{query}::::::::'),
            SeqRecord(Seq(tseq, generic_protein), id=template, name='',
                      description=f'structureX:{template}::{template[5].upper()}::{template[5].upper()}::::')
        ], pir_file, 'pir')
        arg = [self.modpysh, 'python3', Path(__file__).parent.resolve()/'modeller_script.py',
               pir_file, template, query, template_dir]
        subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)


class SWModel:
    def __init__(self, gap_open, gap_extend, modpy_sh_path='/opt/modeller-9.20/bin/modpy.sh'):
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.modpysh = modpy_sh_path

    def generate_protein_model(self, query: str, template: str, alignments: SeqRecord, template_dir: str, out_dir: str):
        ali = pairwise2.align.localds(str(alignments[f'{query}&{template}'].seq.ungap('-')),
                                      str(alignments[f'{template}&{query}'].seq.ungap('-')),
                                      MatrixInfo.blosum62, self.gap_open, self.gap_extend)
        best = ali[np.argmax([_[2] for _ in ali])]
        tseq = replace_missing_residues(str(best[1]), f'{template_dir}/{template}.ent')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pir_file = f'{out_dir}/{template}.pir'
        SeqIO.write([
            SeqRecord(Seq(str(best[0]), generic_protein), id=query, name='',
                      description=f'sequence:{query}::::::::'),
            SeqRecord(Seq(tseq, generic_protein), id=template, name='',
                      description=f'structureX:{template}::{template[5].upper()}::{template[5].upper()}::::')
        ], pir_file, 'pir')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        arg = [self.modpysh, 'python3', Path(__file__).parent.resolve()/'modeller_script.py',
               pir_file, template, query, template_dir]
        subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)


class HHSearchModel:
    def __init__(self, hh_db_dir: str, modpy_sh_path='/opt/modeller-9.20/bin/modpy.sh'):
        self.modpysh = modpy_sh_path
        self.hh_db_dir = hh_db_dir

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
                   '-o', '/dev/null', '-oa3m', f'{query_sid}.a3m', '-cpu', os.cpu_count()]
            subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    def generate_pairwise_alignment(self, query: str, template: str, query_record: SeqRecord, out_dir: str):
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        if not Path(f'{out_dir}/{query}.a3m').exists():
            arg = ['hhsearch', '-i', 'stdin', '-d', f'{self.hh_db_dir}/uniclust30_2018_08',
                   '-o', '/dev/null', '-oa3m', f'{out_dir}/{query}.a3m', '-cpu', str(os.cpu_count())]
            try:
                subprocess.run(arg, input=query_record.format('fasta'),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, check=True)
            except Exception as e:
                print(e)
                raise
        index = {l.split()[0]: (int(l.split()[1]), int(l.split()[2]))
                 for l in Path(f'{self.hh_db_dir}/scop40_a3m.ffindex').read_text().splitlines()}
        if template not in index:
            return
        with Path(f'{self.hh_db_dir}/scop40_a3m.ffdata').open() as f:
            f.seek(index[template][0])
            Path(f'{out_dir}/{template}.a3m').write_text(f.read(index[template][1]))
        arg = ['hhalign', '-glob', '-i', f'{out_dir}/{query}.a3m',
               '-t', f'{out_dir}/{template}.a3m', '-o', f'{out_dir}/{template}.hhr']
        try:
            subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, check=True)
        except Exception as e:
            print(e)
            raise

    def generate_protein_models_from_search(self, query: str, hhr_path: str, out_dir: str, template_dir: str, count=10):
        raw_lines = Path(hhr_path).read_text().splitlines()
        result_points = [i for i, l in enumerate(raw_lines) if l.startswith('No ')]
        created = []
        for i, _ in enumerate(result_points):
            if i == len(result_points)-1:
                lines = raw_lines[result_points[i]:]
            else:
                lines = raw_lines[result_points[i]:result_points[i+1]]
            aln_lines = [l for l in lines if l.startswith('Q ') and 'Consensus' not in l]
            if len(aln_lines) < 1:
                continue
            qseq = Seq(''.join([l.split()[3] for l in aln_lines]), generic_protein)
            template = lines[1].split()[0][1:]
            if template in created:
                continue
            aln_lines = [l for l in lines if l.startswith('T ') and 'Consensus' not in l]
            try:
                tseq = replace_missing_residues(''.join([l.split()[3] for l in aln_lines]),
                                                f'{template_dir}/{template[2:4]}/{template}.ent')
            except Exception as e:
                print(f'wget -O {template[2:4]}/{template}.ent http://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-2.07/{template[2:4]}/{template}.ent')
                continue
            tseq = Seq(tseq, generic_protein)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            pir_file = f'{out_dir}/{template}.pir'
            SeqIO.write([
                SeqRecord(qseq, id=query, name='', description=f'sequence:{query}::::::::'),
                SeqRecord(tseq, id=template, name='',
                          description=f'structureX:{template}::{template[5].upper()}::{template[5].upper()}::::')
            ], pir_file, 'pir')
            arg = [self.modpysh, 'python3', Path(__file__).parent.resolve()/'modeller_script.py',
                   pir_file, template, query, f'{template_dir}/{template[2:4]}']
            try:
                result = subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
            except Exception as e:
                print(e)
                print(result.stdout)
                print(result.stderr)
                continue
            created.append(template)
            if len(created) == count:
                break

    def generate_protein_model(self, query: str, template: str, out_dir: str, template_dir: str):
        if not Path(f'{out_dir}/{template}.hhr').exists():
            return
        raw_lines = Path(f'{out_dir}/{template}.hhr').read_text().splitlines()
        result_points = [i for i, l in enumerate(raw_lines) if l.startswith('No ')]
        assert len(result_points) == 1
        # Get query alignment
        lines = raw_lines[result_points[0]:]
        aln_lines = [l for l in lines if l.startswith('Q ') and 'Consensus' not in l]
        if len(aln_lines) == 0:
            return
        qseq = Seq(''.join([l.split()[3] for l in aln_lines]), generic_protein)
        # Get template alignment
        aln_lines = [l for l in lines if l.startswith('T ') and 'Consensus' not in l]
        tseq = ''.join([l.split()[3] for l in aln_lines])
        tseq = Seq(replace_missing_residues(tseq, f'{template_dir}/{template}.ent'), generic_protein)
        pir_file = f'{out_dir}/{template}.pir'
        SeqIO.write([
            SeqRecord(qseq, id=query, name='', description=f'sequence:{query}::::::::'),
            SeqRecord(tseq, id=template, name='',
                      description=f'structureX:{template}::{template[5].upper()}::{template[5].upper()}::::')
        ], pir_file, 'pir')
        arg = [self.modpysh, 'python3', Path(__file__).parent.resolve()/'modeller_script.py',
               pir_file, template, query, template_dir]
        results = subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
#         print(results.stdout)
#         print(results.stderr)
