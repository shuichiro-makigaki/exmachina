from pathlib import Path

from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import generic_protein
from Bio.Align import MultipleSeqAlignment
import click
import numpy as np

import machina.predict
import machina.generate_alignment
from machina.generate_models import MachinaModel


@click.group(context_settings={'max_content_width': 120})
def main():
    pass


@main.command(help='Generate ASCII PSSM from FASTA file')
@click.option('--query', type=click.Path(exists=True), required=True, help='Query sequence file')
@click.option('--template', type=click.Path(exists=True), required=True, help='Template sequence file')
@click.option('--blastdb', type=click.STRING, default='uniref90', show_default=True, help='BLAST database')
@click.option('--num-iterations', type=click.INT, default=3, show_default=True, help='Iterations of PSI-BLAST')
@click.option('--num-threads', type=click.INT, default=1, show_default=True, help='Multi-threading')
@click.option('--out-dir', type=click.Path(), default='results', show_default=True, help='Output directory')
def generate_profile(query, template, blastdb, num_iterations, num_threads, out_dir):
    o_path = Path(out_dir).expanduser().absolute()
    o_path.mkdir(exist_ok=True, parents=True)
    q_path = Path(query).expanduser().absolute()
    NcbipsiblastCommandline(
        query=q_path, db=blastdb, num_iterations=num_iterations, num_threads=num_threads,
        out_ascii_pssm=o_path/(q_path.stem+'.mtx'), save_pssm_after_last_round=True)()
    q_path = Path(template).expanduser().absolute()
    NcbipsiblastCommandline(
        query=q_path, db=blastdb, num_iterations=num_iterations, num_threads=num_threads,
        out_ascii_pssm=o_path/(q_path.stem+'.mtx'), save_pssm_after_last_round=True)()


@main.command(help='''
Predict substitution score between query and template profiles
''')
@click.option('--query', type=click.Path(exists=True), required=True, help='Query ASCII PSSM')
@click.option('--template', type=click.Path(exists=True), required=True, help='Template ASCII PSSM')
@click.option('--flann-x', type=click.Path(exists=True), show_default=True,
              default='scop40_logscore_tmscore0.5_window5_ratio0.1_x.npy',
              help='Sample data file of FLANN')
@click.option('--flann-y', type=click.Path(exists=True), show_default=True,
              default='scop40_logscore_tmscore0.5_window5_ratio0.1_y.npy',
              help='Sample label file of FLANN')
@click.option('--flann-index', type=click.Path(exists=True), show_default=True,
              default='flann19_scop40_logscore_tmscore0.5_window5_ratio0.1',
              help='Index file of FLANN')
@click.option('--num-neighbors', type=int, default=1000, show_default=True,
              help='Number of neighbors for kNN')
@click.option('--out-dir', type=click.Path(), default='results', show_default=True,
              help='Output directory for results')
@click.option('--out-name', type=click.Path(), default='score.npy', show_default=True,
              help='Output file name in --out-dir')
def predict_scores(query, template, flann_x, flann_y, flann_index, num_neighbors, out_dir, out_name):
    machina.predict.predict_scores(
        Path(query), Path(template),
        flann_x=Path(flann_x), flann_y=Path(flann_y), flann_index=Path(flann_index),
        num_neighbors=num_neighbors, out_dir=Path(out_dir), out_name=Path(out_name))


@main.command(help='Generate alignment from predicted substitution score matrix')
@click.option('--query', type=click.Path(exists=True), required=True, help='Query ASCII PSSM')
@click.option('--template', type=click.Path(exists=True), required=True, help='Template ASCII PSSM')
@click.option('--score-matrix', type=click.Path(exists=True), required=True, help='Predicted score matrix')
@click.option('--open-penalty', type=click.FLOAT, default=0.1, show_default=True,
              help='Smith-Waterman gap-open penalty')
@click.option('--extend-penalty', type=click.FLOAT, default=0.0001, show_default=True,
              help='Smith-Waterman gap-extend penalty')
@click.option('--out-dir', type=click.Path(), default='results', show_default=True,
              help='Output directory for results')
@click.option('--out-name', type=click.Path(), default='alignments.npy', show_default=True,
              help='Output file name in --out-dir')
def generate_alignment(query, template, score_matrix, open_penalty, extend_penalty, out_dir, out_name):
    machina.generate_alignment.alignment_local_and_save(
        Path(score_matrix), Path(query), Path(template), -open_penalty, -extend_penalty, Path(out_dir), Path(out_name))


@main.command(help='Show alignments')
@click.option('--ali-path', type=click.Path(exists=True), required=True, help='Alignment result file')
@click.option('--out-format', type=click.Choice(['fasta', 'pir', 'clustal']), default='clustal',
              show_default=True, help='Alignment format')
def show_alignments(ali_path, out_format):
    for aln in np.load(ali_path, allow_pickle=True):
        if out_format == 'pir':
            msa = MultipleSeqAlignment([
                SeqRecord(Seq(aln[0], generic_protein), id='Query', name='',
                          description='sequence:::::::::'),
                SeqRecord(Seq(aln[1], generic_protein), id='Template', name='',
                          description='structureX:::::::::')
            ])
        else:
            msa = MultipleSeqAlignment([
                SeqRecord(Seq(aln[0], generic_protein), id='Query', name='', description=''),
                SeqRecord(Seq(aln[1], generic_protein), id='Template', name='', description='')
            ])
        print(msa.format(out_format))


@main.command(help='Generate protein model by MODELLER')
@click.argument('modeller_bin', type=click.Path(exists=True))
@click.argument('query', type=click.Path(exists=True))
@click.argument('template', type=click.Path(exists=True))
@click.argument('chain', type=str)
@click.argument('alignment', type=click.Path(exists=True))
@click.argument('atom_dir', type=click.Path(exists=True))
@click.option('--out-dir', type=click.Path(), default='results')
def generate_model(modeller_bin, query, template, chain, alignment, atom_dir, out_dir):
    MachinaModel(mod_bin=modeller_bin).generate_protein_model(
        Path(query), Path(template), chain, Path(alignment), Path(atom_dir), Path(out_dir))


def download_knn_model():
    # ToDo
    pass


if __name__ == '__main__':
    main()
