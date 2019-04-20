# exmachina

Sequence aligner for accurate template-based protein structure prediction

## Supported system

Linux and Python **>=3.6**

## Prepare

### 1. Install requirements

```bash
pip3 install -r requirements.txt
```

We also needs FLANN and the Python binding.

```bash
mkdir -p ~/.local/src
cd ~/.local/src
git clone https://github.com/mariusmuja/flann
cd flann
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/.local .. 
```

### 2. Download <i>k</i>NN index tree and training data

```bash
wget -O data/train/flann19_scop40_logscore_tmscore0.5_window5_ratio0.1 http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/flann19_scop40_logscore_tmscore0.5_window5_ratio0.1
wget -O data/train/scop40_logscore_tmscore0.5_window5_ratio0.1_x.npy http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/scop40_logscore_tmscore0.5_window5_ratio0.1_x.npy
wget -O data/train/scop40_logscore_tmscore0.5_window5_ratio0.1_y.npy http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/scop40_logscore_tmscore0.5_window5_ratio0.1_y.npy
```

## How to use in `example.ipynb`

Some notebook extensions are recommended to visualize results.

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install nglview-js-widgets
```

Run jupyter notebook,

```bash
jupyter lab
```

and open `example.ipynb`.
