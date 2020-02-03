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

## Prepare

### 1. Install requirements

```shell script
pip3 install -r requirements.txt
```

We also need FLANN and Python binding.

```shell script
git clone https://github.com/mariusmuja/flann
cd flann
### These two procedures are needed only for MacOSX
# brew install cmake hdf5
# git checkout 033b05b1 -b clang
mkdir build
cd build
### PYTHON_EXECUTABLE:FILEPATH and CMAKE_INSTALL_PREFIX should follow your environment.
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

### 2. Download <i>k</i>NN index tree and training data

This consumes 24GB RAM.

```bash
wget http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/flann19_scop40_logscore_tmscore0.5_window5_ratio0.1
wget http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/scop40_logscore_tmscore0.5_window5_ratio0.1_x.npy
wget http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/scop40_logscore_tmscore0.5_window5_ratio0.1_y.npy
```

The smaller dataset is available for test.

```bash
wget http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/flann19_scop40_logscore_tmscore0.5_window5_ratio0.01
wget http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/scop40_logscore_tmscore0.5_window5_ratio0.01_x.npy
wget http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/scop40_logscore_tmscore0.5_window5_ratio0.01_y.npy
```

### 3. Open example.ipynb in Jupyter Lab.