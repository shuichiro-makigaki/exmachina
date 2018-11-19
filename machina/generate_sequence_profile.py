import logging
from pathlib import Path
import os
import pickle

from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Blast.Applications import NcbipsiblastCommandline
from tqdm import tqdm

from TMtoolsCommandLine import TMalignCommandLine

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


def main():
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


if __name__ == '__main__':
    main()
