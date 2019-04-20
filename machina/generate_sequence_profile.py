import logging
from pathlib import Path
import os
import pickle

from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Blast.Applications import NcbipsiblastCommandline
from tqdm import tqdm

from .TMtoolsCommandLine import TMalignCommandLine


def generate_sequence_profiles(structural_alignment_path, pssm_dir, blastdb='uniref90'):
    results = []
    seq_index = SeqIO.index(structural_alignment_path, 'fasta')
    for record_id in seq_index:
        results.append(record_id.split('&')[0])
        results.append(record_id.split('&')[1])

    for domain in tqdm(set(results)):
        if Path(f'{pssm_dir}/{domain[2:4]}/{domain}.mtx').exists():
            continue
        Path(f'{pssm_dir}/{domain[2:4]}').mkdir(parents=True, exist_ok=True)
        seq_record = seq_index[[_ for _ in seq_index if _.startswith(f'{domain}&')][0]]
        NcbipsiblastCommandline(db=blastdb, num_threads=os.cpu_count(), num_iterations=3,
                                out_ascii_pssm=Path(f'{pssm_dir}/{domain[2:4]}/{domain}.mtx').as_posix(),
                                save_pssm_after_last_round=True
                                )(stdin=str(seq_record.seq))


def __generate_sequence_profiles_old():
    mtx_dir_name = 'pssm_deltablast'
    DB_INDEX = SeqIO.index('data/scop40_structural_alignment.fasta', 'fasta')
    records = {}
    for i in DB_INDEX:
        domkey = i.split('&')[0]
        records[domkey] = SeqRecord(DB_INDEX[i].seq.ungap('-'), id=domkey, name='', description='')
    with Path('data/scop40_scopdom_pdbatom_seq.fasta').open('w') as f:
        SeqIO.write(records.values(), f, 'fasta')

    DB_INDEX = SeqIO.index('data/scop40_scopdom_pdbatom_seq.fasta', 'fasta')
    for sid in tqdm(list(DB_INDEX)):
        mtx_dir = Path(f'data/{mtx_dir_name}/{sid[2:4]}')
        mtx_dir.mkdir(exist_ok=True, parents=True)
        mtx_file = mtx_dir/f'{sid}.mtx'
        if mtx_file.exists():
            logging.debug(f'PSSM already exists: {mtx_file}')
            continue
        try:
            SeqIO.write(DB_INDEX[sid], f'{sid}.fasta', 'fasta')
            NcbipsiblastCommandline(query=f'{sid}.fasta', db='uniref90', num_threads=int(os.cpu_count()),
                                    num_iterations=3, out_ascii_pssm=mtx_file.as_posix(),
                                    save_pssm_after_last_round=True)()
        except Exception as e:
            logging.exception(e)
            continue
        finally:
            if Path(f'{sid}.fasta').exists():
                Path(f'{sid}.fasta').unlink()

    logging.info('')
    for sid in tqdm(pickle.load(Path('data/one_domain_superfamily.pkl').open('rb'))):
        mtx_dir = Path(f'data/{mtx_dir_name}/{sid[2:4]}')
        mtx_dir.mkdir(exist_ok=True, parents=True)
        mtx_file = mtx_dir/f'{sid}.mtx'
        if mtx_file.exists():
            logging.debug(f'PSSM already exists: {mtx_file}')
            continue
        try:
            tmalign = TMalignCommandLine(f'data/scop_e/{sid[2:4]}/{sid}.ent', f'data/scop_e/{sid[2:4]}/{sid}.ent')
            tmalign.run()
            assert str(tmalign.alignment[0].seq).find('-') == -1
            SeqIO.write(tmalign.alignment[0], f'{sid}.fasta', 'fasta')
            NcbipsiblastCommandline(query=f'{sid}.fasta', db='uniref90', num_threads=int(os.cpu_count()),
                                    num_iterations=3, out_ascii_pssm=mtx_file.as_posix(),
                                    save_pssm_after_last_round=True)()
        except Exception as e:
            logging.error(f'sid={sid}')
            logging.exception(e)
            continue
        finally:
            if Path(f'{sid}.fasta').exists():
                Path(f'{sid}.fasta').unlink()
