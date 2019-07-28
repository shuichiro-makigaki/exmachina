from pathlib import Path

from Bio.Blast.Applications import NcbipsiblastCommandline
import click
import numpy as np

import machina.predict
import machina.generate_alignment
from machina.generate_models import MachinaModel


@click.group()
def main():
    pass


@main.command(help='''
Create ASCII PSSM from query sequence file (FASTA)

QUERY: Query sequence file in .fasta

TEMPLATE: Template sequence file in .fasta
''')
@click.argument('query', type=click.Path(exists=True))
@click.argument('template', type=click.Path(exists=True))
@click.option('--cmd', type=str, default='psiblast', help='BLAST command (Default: psiblast, recommended)')
@click.option('--blastdb', type=str, default='uniref90',
              help='DB name in $BLASTDB or absolute path (Default: uniref90, recommended)')
@click.option('--num-iterations', type=int, default=3,
              help='Number of search iterations only for PSI-BLAST (Default: 3)')
@click.option('--out-dir', type=click.Path(), default='results')
def create_profile(query, template, blastdb, num_iterations, cmd, out_dir):
    o_path = Path(out_dir).expanduser().absolute()
    o_path.mkdir(exist_ok=True, parents=True)
    q_path = Path(query).expanduser().absolute()
    NcbipsiblastCommandline(
        query=q_path, db=blastdb,
        num_iterations=num_iterations,
        out_ascii_pssm=o_path/(q_path.stem+'.mtx'),
        save_pssm_after_last_round=True)()
    q_path = Path(template).expanduser().absolute()
    NcbipsiblastCommandline(
        query=q_path, db=blastdb,
        num_iterations=num_iterations,
        out_ascii_pssm=o_path/(q_path.stem+'.mtx'),
        save_pssm_after_last_round=True)()


@main.command(help='''
Predict substitution score between query and template profile

QUERY: Query profile file in ASCII PSSM

TEMPLATE: Template profile file in ASCII PSSM
''')
@click.argument('query', type=click.Path(exists=True))
@click.argument('template', type=click.Path(exists=True))
@click.option('--flann-x', type=click.Path(exists=True),
              default='scop40_logscore_tmscore0.5_window5_ratio0.1_x.npy',
              help='Sample data file of FLANN (Default: scop40_logscore_tmscore0.5_window5_ratio0.1_x.npy)')
@click.option('--flann-y', type=click.Path(exists=True),
              default='scop40_logscore_tmscore0.5_window5_ratio0.1_y.npy',
              help='Sample label file of FLANN (Default: scop40_logscore_tmscore0.5_window5_ratio0.1_y.npy)')
@click.option('--flann-index', type=click.Path(exists=True),
              default='flann19_scop40_logscore_tmscore0.5_window5_ratio0.1',
              help='Index file of FLANN (Default: flann19_scop40_logscore_tmscore0.5_window5_ratio0.1)')
@click.option('--num-neighbors', type=int, default=1000,
              help='Number of neighbors for kNN (Default: 1000)')
@click.option('--out-dir', type=click.Path(), default='results',
              help='Output directory for results (Default: results)')
@click.option('--out-name', type=click.Path(), default='score.npy',
              help='Output file name in --out-dir (Default: score.npy)')
def predict_scores(query, template, flann_x, flann_y, flann_index, num_neighbors, out_dir, out_name):
    machina.predict.predict_scores(
        Path(query), Path(template),
        flann_x=Path(flann_x), flann_y=Path(flann_y), flann_index=Path(flann_index),
        num_neighbors=num_neighbors, out_dir=Path(out_dir), out_name=Path(out_name))


@main.command(help='Generate alignment from predicted substitution score matrix')
@click.argument('query', type=click.Path(exists=True))
@click.argument('template', type=click.Path(exists=True))
@click.argument('score_matrix', type=click.Path(exists=True))
@click.option('--open-penalty', type=float, default=0.1)
@click.option('--extend-penalty', type=float, default=0.0001)
@click.option('--out-dir', type=click.Path(), default='results')
@click.option('--out-name', type=click.Path(), default='alignments.npy')
def generate_alignment(query, template, score_matrix, open_penalty, extend_penalty, out_dir, out_name):
    machina.generate_alignment.alignment_local_and_save(
        Path(score_matrix), Path(query), Path(template), -open_penalty, -extend_penalty, Path(out_dir), Path(out_name))


# @main.command(help='Show alignments')
# @click.argument('path', type=click.Path(exists=True))
# @click.option('--out-format', type=click.Choice(['fasta', 'pir']), default='fasta')
def show_alignment(path, out_format):
    # ToDo
    for aln in np.load(path, allow_pickle=True):
        print(f'> alignment_1 Score={aln[2]}')
        print(aln[0])
        print(aln[1])
        print('')


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
