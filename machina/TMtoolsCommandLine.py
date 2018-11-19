import sys
import subprocess
import logging
from pathlib import Path
import re

from Bio.PDB import PDBParser, PPBuilder, PDBIO
from Bio.Alphabet import generic_protein
from Bio.PDB.Chain import Chain
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.AlignIO import MultipleSeqAlignment

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


class TMscoreCommandLine:
    def __init__(self, protein_a=None, protein_b=None, binary='TMscore'):
        self.protein_A = protein_a
        self.protein_B = protein_b
        self.cmd = binary
        self.stdout = None
        self.stderr = None
        self.alignment = None
        self.tmscore = None
        self.maxsub = None
        self.gdtts = None
        self.gdtha = None
        self.rmsd = None
        self.num_res_in_common = None
        self.len_a = None
        self.len_b = None

    def _pp(self, pdb_path, chain_id):
        pdb_id = Path(pdb_path).stem
        pp_list = PPBuilder().build_peptides(PDBParser().get_structure(pdb_id, pdb_path)[0][chain_id])
        pp = pp_list[0]
        for i in pp_list[1:]:
            pp += i
        return pp

    def _align(self):
        pp_a = self._pp(self.protein_A, 'A')
        # seq_a = pp_a.get_sequence()
        pp_b = self._pp(self.protein_B, ' ')
        # seq_b = pp_b.get_sequence()

        # global_align = pairwise2.align.globalxx(seq_a, seq_b)[0]
        # msa = MultipleSeqAlignment([SeqRecord(Seq(global_align[0], alphabet=generic_protein), id='A'),
        #                             SeqRecord(Seq(global_align[1], alphabet=generic_protein), id='B')])
        msa = self.alignment
        # offset_a = re.search(r'[^-]', str(msa[0].seq)).span()[0]
        # offset_b = re.search(r'[^-]', str(msa[1].seq)).span()[0]
        plus = 1000
        for i in range(len(pp_a)):
            pp_a[i].id = (pp_a[i].id[0], plus + i, pp_a[i].id[2])
        for i in range(len(pp_b)):
            pp_b[i].id = (pp_b[i].id[0], plus + i, pp_b[i].id[2])
        new_chain_a = Chain(' ')
        for i in pp_a:
            # i.id = (i.id[0], i.id[1] - plus, i.id[2])
            new_chain_a.add(i)
        new_chain_b = Chain(' ')
        for i in pp_b:
            # i.id = (i.id[0], i.id[1] - plus, i.id[2])
            new_chain_b.add(i)

        io = PDBIO()
        io.set_structure(new_chain_a)
        io.save(f'.tmp.protein_a.pdb')
        io = PDBIO()
        io.set_structure(new_chain_b)
        io.save(f'.tmp.protein_b.pdb')

    def _parse(self):
        for i, l in enumerate(self.stdout):
            if re.match('^\(":" denotes', l):
                a = SeqRecord(Seq(self.stdout[i+1], alphabet=generic_protein), id='Protein_A')
                b = SeqRecord(Seq(self.stdout[i+3], alphabet=generic_protein), id='Protein_B')
                self.alignment = MultipleSeqAlignment([a, b])
                break
        for l in self.stdout:
            if re.match('^TM-score', l):
                self.tmscore = float(l.split('=')[1].split('(')[0].replace(' ', ''))
                break
        for l in self.stdout:
            if re.match('^MaxSub-score', l):
                self.maxsub = float(l.split('=')[1].split('(')[0].replace(' ', ''))
                break
        for l in self.stdout:
            if re.match('^GDT-TS-score', l):
                self.gdtts = float(l.split('=')[1].split('%')[0].replace(' ', ''))
                break
        for l in self.stdout:
            if re.match('^GDT-HA-score', l):
                self.gdtha = float(l.split('=')[1].split('%')[0].replace(' ', ''))
                break
        for l in self.stdout:
            if re.match('^RMSD of', l):
                self.rmsd = float(l.split('=')[1].replace(' ', ''))
                break
        for l in self.stdout:
            if re.match('^Number of residues in common', l):
                self.num_res_in_common = int(l.split('=')[1].replace(' ', ''))
                break
        for l in self.stdout:
            if re.match('^Structure1: ', l):
                self.len_a = int(l.split('=')[1].split('(')[0].replace(' ', ''))
                break
        for l in self.stdout:
            if re.match('^Structure2: ', l):
                self.len_b = int(l.split('=')[1].split('(')[0].replace(' ', ''))
                break

    def _log(self):
        for l in self.stdout:
            logging.info(l)
        for l in self.stderr:
            logging.error(l)

    def run(self, print_log=False):
        arg = [self.cmd, self.protein_A, self.protein_B]
        p = subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.stdout = p.stdout.splitlines()
        self.stderr = p.stderr.splitlines()
        if print_log:
            self._log()
        self._parse()
        self._align()
        arg = [self.cmd, '.tmp.protein_a.pdb', '.tmp.protein_b.pdb']
        p = subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.stdout = p.stdout.splitlines()
        self.stderr = p.stderr.splitlines()
        if print_log:
            self._log()
        self._parse()


class TMalignCommandLine:
    def __init__(self, protein_a=None, protein_b=None, binary='TMalign'):
        self.protein_A = protein_a
        self.protein_B = protein_b
        self.cmd = binary
        self.tmscore = None
        self.stdout = None
        self.stderr = None
        self.alignment = None

    def _parse(self):
        for i, l in enumerate(self.stdout):
            if re.match('^TM-score=', l):
                self.tmscore = (float(self.stdout[i].split(' ')[1]), float(self.stdout[i+1].split(' ')[1]))
                break
        for i, l in enumerate(self.stdout):
            if re.match('^\(":" denotes', l):
                a = SeqRecord(Seq(self.stdout[i+1], alphabet=generic_protein),
                              id=Path(self.protein_A).stem + '&' + Path(self.protein_B).stem,
                              description=f'TM-score={self.tmscore[0]}')
                b = SeqRecord(Seq(self.stdout[i+3], alphabet=generic_protein),
                              id=Path(self.protein_B).stem + '&' + Path(self.protein_A).stem,
                              description=f'TM-score={self.tmscore[1]}')
                self.alignment = MultipleSeqAlignment([a, b])
                break

    def _log(self):
        for l in self.stdout:
            logging.info(l)
        for l in self.stderr:
            logging.error(l)

    def run(self, print_log=False):
        arg = [self.cmd, self.protein_A, self.protein_B]
        p = subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.stdout = p.stdout.splitlines()
        self.stderr = p.stderr.splitlines()
        if print_log:
            self._log()
        self._parse()


def run_tmscore():
    native = 'data/scop_e/os/d1osna_.ent'
    model = 'results/1osna_-1w5sa2.pdb'
    runner = TMscoreCommandLine(native, model)
    runner.run()


def main():
    if sys.argv[1] == 'TMscore':
        run_tmscore()


if __name__ == '__main__':
    main()
