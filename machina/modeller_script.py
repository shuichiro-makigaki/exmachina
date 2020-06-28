import os
import sys
import glob
import shutil

from modeller import *
from modeller.automodel import *


def main():
    pir_path = sys.argv[1]
    query_id = sys.argv[2]
    template_id = sys.argv[3]
    atom_files_dir = sys.argv[4]
    conv_modres = False
    if len(sys.argv) == 6 and sys.argv[5] in ['true']:
        conv_modres = True

    env = environ()
    env.io.atom_files_directory.append(atom_files_dir)
    env.io.convert_modres = conv_modres

    auto = automodel(env, alnfile=pir_path, knowns=[template_id], sequence=query_id)
    auto.make()

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
    for f in glob.glob(query_id + ".B*.pdb"):
        shutil.copy(f, os.path.dirname(pir_path) + '/' + query_id + '.pdb')
        os.remove(f)
    for f in glob.glob(os.path.splitext(__file__)[0] + '.log'):
        shutil.copy(f, os.path.curdir + '/modeller.log')
        os.remove(f)


if __name__ == "__main__":
    main()
