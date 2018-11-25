# machina

Sequence aligner for accurate template-based protein structure prediction

## Supported system

Linux and Python **>=3.6**

## Prepare

### 1. Install requirements

```bash
pip3 install -r requirements.txt
```

### 2. Download <i>k</i>NN index tree and training data

```bash
wget -O data/train/flann19_scop40_logscore_tmscore50_w5_randomsampling_ratio0.1 http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/flann19_scop40_logscore_tmscore50_w5_randomsampling_ratio0.1
wget -O data/train/scop40_logscore_tmscore50_w5_randomsampling_ratio0.1_x.npy http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/scop40_logscore_tmscore50_w5_randomsampling_ratio0.1_x.npy
wget -O data/train/scop40_logscore_tmscore50_w5_randomsampling_ratio0.1_y.npy http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/scop40_logscore_tmscore50_w5_randomsampling_ratio0.1_y.npy
```

## How to use in `example.ipynb`

Some notebook extensions are recommended to visualize results.

```bash
jupyter nbextension enable widgetsnbextension --py
jupyter nbextension enable nglview --py
```

Run jupyter notebook,

```bash
jupyter notebook
```

and open `example.ipynb`.
