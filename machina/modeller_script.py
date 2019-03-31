import os
import sys
import glob
import logging
import shutil

from modeller import *
from modeller.automodel import *

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


def modeller_automodel(pir_path, template_id, query_id, atom_files_dir):
    env = environ()
    env.io.atom_files_directory.append(atom_files_dir)
    # env.io.convert_modres = False

    try:
        auto = automodel(env, alnfile=pir_path, knowns=[template_id], sequence=query_id)
        auto.make()
    except Exception as e:
        logging.info(e)

    for f in glob.glob(query_id + ".B*.pdb"):
        shutil.copy(f, os.path.splitext(pir_path)[0] + ".pdb")
        os.remove(f)
    for f in glob.glob(query_id + ".V*"):
        os.remove(f)
    for f in glob.glob(query_id + ".D*"):
        os.remove(f)
    for f in glob.glob(query_id + ".ini"):
        os.remove(f)
    for f in glob.glob(query_id + ".rsr"):
        os.remove(f)
    for f in glob.glob(query_id + ".sch"):
        os.remove(f)
    for f in glob.glob(os.path.splitext(__file__)[0] + '.log'):
        shutil.copy(f, os.path.splitext(pir_path)[0] + '.log')
        os.remove(f)


def main():
    modeller_automodel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
