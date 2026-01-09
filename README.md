# Systematic discovery of single-cell protein networks in cancer with Shusi

![Shusi](fig_data/shusi.png)

This repository provides **Shusi**, a graph-based framework for systematic discovery of **single-cell protein networks in cancer**. It includes a ready-to-run inference pipeline driven by a single YAML config (`infer_config.yaml`).

## Quickstart (Inference)

### 1) Setup

Create a Python environment (Python 3.9+ recommended), then install dependencies.

At minimum you will need:
- `torch`
- `torch-geometric`
- `numpy`, `scipy`

> Note: PyTorch Geometric wheels depend on your PyTorch/CUDA version. Please install `torch` first, then install `torch-geometric` accordingly.

### 2) Download pretrained artifacts (not tracked in git)

The following large files are hosted on Google Drive:

- **Model checkpoint** (`checkpoint/shusi.pth`)  
  https://drive.google.com/file/d/1KjIViKOyKaai9cq_yKCKhowmhlXtXNYV/view?usp=drive_link

- **Feature maps**  
  `data/feat_data/embedding_anno_gene.pkl`  
  https://drive.google.com/file/d/1DqXEXzlcHMjg_BddiuhZx9FwC-6JWRo4/view?usp=drive_link  
  `data/feat_data/embedding_anno_sentence.pkl`  
  https://drive.google.com/file/d/1zM9LrHwDwIxgqTNnssl-30JLzckb0Dyy/view?usp=drive_link

After downloading, place them **exactly** at the paths used by the config:

```bash
mkdir -p checkpoint data/feat_data
```

```text
checkpoint/
  shusi.pth
data/
  feat_data/
    embedding_anno_gene.pkl
    embedding_anno_sentence.pkl
```

If you prefer different locations, update the corresponding fields in `infer_config.yaml`:
- `model.ckpt`
- `feat.feat_map1`
- `feat.feat_map2`

### 3) Configure your input graph

Edit `infer_config.yaml` and set either:
- **Single sample** (recommended):
  - `cell.x_path`: path to `*_x.npy`
  - `cell.edge_path`: path to `*_edge.npz`
- **Batch mode**:
  - `cell.dir`: a directory containing multiple `*_x.npy` + `*_edge.npz` pairs
  - `cell.recursive`: whether to search subfolders

### 4) Run

```bash
python inference.py --config infer_config.yaml
```

To generate a predicted-edge CSV, set `mode: predict_all` in `infer_config.yaml`.

## Outputs

The behavior depends on `mode` in `infer_config.yaml`:

- `mode: eval_masked`  
  Masks a ratio of edges and prints `precision@k` to stdout.

- `mode: predict_all`  
  Predicts novel edges and writes a CSV like `inference_outputs/pred_edges.csv` (or per-sample files under `output.edges_dir` if set).

The prediction CSV uses the schema:

```text
gene1,gene2,score
```

## Notes

- If `device: cuda:0` is set but CUDA is unavailable, the code automatically falls back to CPU.
- Relative paths in `infer_config.yaml` are resolved from the repository root.
- `gene_map` is optional; if it is missing, outputs fall back to gene IDs.
