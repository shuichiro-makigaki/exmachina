import itertools
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from Bio.SCOP import Scop
from Bio import SeqIO

from TMtoolsCommandLine import TMalignCommandLine


def run_tmalign_async(dom1, dom2):
    try:
        tmalign = TMalignCommandLine(dom1, dom2)
        tmalign.run()
        return tmalign.alignment
    except Exception as e:
        logging.exception(e)
    return None


def generate_structural_alignments(scop_dir: str, scop_version: str, pdb_dir: str, out_file: str):
    scop40 = SeqIO.index('data/train/astral-scopdom-seqres-gd-sel-gs-bib-40-1.75.fa', 'fasta')
    scop_root = Scop(dir_path=scop_dir, version=scop_version).getRoot()
    results = []
    for cl in scop_root.getChildren():
        for cf in cl.getChildren():
            for sf in cf.getChildren():
                px = sf.getDescendents('px')
                if len(px) < 2:
                    continue
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for c in itertools.combinations(px, 2):
                        if c[0].sid in scop40 and c[1].sid in scop40:
                            futures.append(executor.submit(run_tmalign_async,
                                                           f'{pdb_dir}/{c[0].sid[2:4]}/{c[0].sid}.ent',
                                                           f'{pdb_dir}/{c[1].sid[2:4]}/{c[1].sid}.ent'))
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            results.append(result[0])
                            results.append(result[1])
    SeqIO.write(results, out_file, 'fasta')
