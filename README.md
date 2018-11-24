# machina

Sequence aligner for accurate template-based protein structure prediction

## Supported system

Linux and Python **>=3.6**

## Prepare

Install requirements.

```
pip3 install -r requirements.txt
```

Download kNN index tree and data.

```bash
wget -O data/train/flann19_scop40_logscore_tmscore50_w5_randomsampling_ratio0.1 http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/flann19_scop40_logscore_tmscore50_w5_randomsampling_ratio0.1
wget -O data/train/scop40_logscore_tmscore50_w5_randomsampling_ratio0.1_x.npy http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/scop40_logscore_tmscore50_w5_randomsampling_ratio0.1_x.npy
wget -O data/train/scop40_logscore_tmscore50_w5_randomsampling_ratio0.1_y.npy http://www.cb.cs.titech.ac.jp/~makigaki/machina/data/train/scop40_logscore_tmscore50_w5_randomsampling_ratio0.1_y.npy
```

Some notebook extensions are required.

```bash
jupyter nbextension enable widgetsnbextension --py
jupyter nbextension enable nglview --py
```

## How to use in `example.ipynb`

Run

```bash
jupyter notebook
```

Open `example.ipynb`.
