# ExMachina

Sequence aligner for accurate template-based protein structure prediction

---

Template-based modeling, the process of predicting the tertiary structure of a protein by using homologous protein structures, is useful if good templates can be found. Although modern homology detection methods can find remote homologs with high sensitivity, the accuracy of template-based models generated from homology-detection-based alignments is often lower than that from ideal alignments.

We propose a new method that generates pairwise sequence alignments for more accurate template-based modeling. The proposed method trains a machine learning model using the structural alignment of known homologs. When calculating sequence alignments, instead of a fixed substitution matrix, this method dynamically predicts a substitution score from the trained model.

```
@article{10.1093/bioinformatics/btz483,
    author = {Makigaki, Shuichiro and Ishida, Takashi},
    title = "{Sequence alignment using machine learning for accurate template-based protein structure prediction}",
    journal = {Bioinformatics},
    year = {2019},
    month = {06},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btz483},
    url = {https://doi.org/10.1093/bioinformatics/btz483},
}
```

## Supported system

* Linux (recommended) and macOS X
* Python **>=3.6**

## How to use

Open and read `example.ipynb`.

### Available in Docker

![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/makisyu/exmachina)

### Build from source

Install requirements (use `--user` for non-root installation):

```shell script
pip3 install [--user] -r requirements.txt
```

We also need FLANN [1] and Python binding.

```shell script
git clone https://github.com/mariusmuja/flann
cd flann
### These two procedures are needed only for MacOSX
# brew install cmake hdf5
# git checkout 033b05b1 -b clang
mkdir build
cd build
### Environment variables `PYTHON_EXECUTABLE:FILEPATH` and `CMAKE_INSTALL_PREFIX` should follow your environment.
### liblz4 is required.
cmake -DCMAKE_INSTALL_PREFIX=~/.local -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) ..
make
make install
```

For non-root (`--user`) installation, edit `flann/src/python/CMakeLists.txt`:

```shell script
# Around #10
COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py install
# Add --user
COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py install --user
```

## Supplementary Data

Data used for the kNN index and the paper is available by `dvc pull data.dvc`, which requires 140GB disk space.

## References

1. Marius Muja and David G. Lowe, "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration", in <i>International Conference on Computer Vision Theory and Applications (VISAPP'09)</i>, 2009
