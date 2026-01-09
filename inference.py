"""
YAML-configured inference script.

Supports 2 modes in `infer_config.yaml`:
- `mode: predict_all`: encode with all observed edges, then predict top-k new edges.
- `mode: eval_masked`: mask a ratio of edges, predict candidates, and report precision@k (like legacy `inference.py`).

This file intentionally does NOT import `inference_for_eval.py`, but reimplements the same output/write behavior.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges

from model import GNN_Backbone, VGAEModel
from utils import convert_to_directed_edge, get_data


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "infer_config.yaml"
X_SUFFIX = "_x.npy"
EDGE_SUFFIX = "_edge.npz"


def set_seed(seed: int) -> None:
    import random

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Load YAML using PyYAML if available. Falls back to a minimal parser for:
      - `key: value`
      - nested dict via 2-space indentation
      - inline lists: `[1, 2, 3]`
    """
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping/dict.")
        return data
    except ModuleNotFoundError:
        pass

    def parse_scalar(raw: str) -> Any:
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            if not inner:
                return []
            return [parse_scalar(x.strip()) for x in inner.split(",")]
        if raw.lower() in {"true", "false"}:
            return raw.lower() == "true"
        if raw.lower() in {"null", "none", "~"}:
            return None
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            return raw.strip("\"'")

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, root)]

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"Unsupported indentation (must be 2 spaces): {line!r}")
        key, sep, rest = line.strip().partition(":")
        if sep != ":":
            raise ValueError(f"Invalid YAML line: {line!r}")
        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Bad indentation: {line!r}")
        current = stack[-1][1]
        if rest.strip() == "":
            new_dict: dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent + 2, new_dict))
        else:
            current[key] = parse_scalar(rest)
    return root


def _get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_str_or_none(v: Any) -> str | None:
    if v in (None, "null", "None", ""):
        return None
    return str(v)


@dataclass(frozen=True)
class InferConfig:
    seed: int
    device: str
    mode: str

    cell_prefix: str | None
    cell_x_path: str | None
    cell_edge_path: str | None
    cell_dir: str | None
    cell_recursive: bool
    cell_max_samples: int | None

    model_name: str
    model_ckpt: str
    num_layers: int
    hid_dim: int
    edge_attr: int
    model_size: int

    feat_type: str
    feat_map1: str
    feat_map2: str
    gene_map: str | None

    masked_test_ratio: float
    masked_precision_at: list[int]
    masked_oversample_factor: int
    masked_topk_candidates: int

    topk: int
    oversample: int
    normalize_scores: bool

    output_csv: str
    output_metrics_csv: str | None
    output_eval_csv: str | None
    output_edges_dir: str | None


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> InferConfig:
    cfg = _load_yaml(path)

    seed = int(_get(cfg, "seed", 42))
    device = str(_get(cfg, "device", "cpu"))
    mode = str(_get(cfg, "mode", "predict_all"))
    if mode not in {"predict_all", "eval_masked"}:
        raise ValueError(f"Unsupported mode={mode!r}. Use 'predict_all' or 'eval_masked'.")

    cell_prefix = _as_str_or_none(_get(cfg, "cell.prefix"))
    cell_x_path = _as_str_or_none(_get(cfg, "cell.x_path"))
    cell_edge_path = _as_str_or_none(_get(cfg, "cell.edge_path"))
    cell_dir = _as_str_or_none(_get(cfg, "cell.dir"))
    cell_recursive = bool(_get(cfg, "cell.recursive", False))
    cell_max_samples = _get(cfg, "cell.max_samples")
    cell_max_samples = None if cell_max_samples in (None, "null", "None", "") else int(cell_max_samples)

    model_name = str(_get(cfg, "model.name", "vgae_gin"))
    model_ckpt = str(_get(cfg, "model.ckpt"))
    if not model_ckpt:
        raise ValueError("Missing required config: model.ckpt")

    num_layers = int(_get(cfg, "model.num_layers", 2))
    hid_dim = int(_get(cfg, "model.hid_dim", 512))
    edge_attr = int(_get(cfg, "model.edge_attr", 1))
    model_size = int(_get(cfg, "model.size", _get(cfg, "model_size", 50000)))

    feat_type = str(_get(cfg, "feat.type", "all"))
    feat_map1 = str(_get(cfg, "feat.feat_map1", "data/feat_data/embedding_anno_gene.pkl"))
    feat_map2 = str(_get(cfg, "feat.feat_map2", "data/feat_data/embedding_anno_sentence.pkl"))
    gene_map = _as_str_or_none(_get(cfg, "gene_map", "data/mapping_data/gene_mapping_all.pkl"))

    masked_test_ratio = float(_get(cfg, "masked_eval.test_ratio", 0.2))
    precision_at_val = _get(cfg, "masked_eval.precision_at", [1000, 10000, 20000, 50000])
    if isinstance(precision_at_val, str):
        masked_precision_at = [int(x.strip()) for x in precision_at_val.split(",") if x.strip()]
    elif isinstance(precision_at_val, list):
        masked_precision_at = [int(x) for x in precision_at_val]
    else:
        masked_precision_at = [1000, 10000, 20000, 50000]

    masked_oversample_factor = int(_get(cfg, "masked_eval.oversample_factor", 10))
    masked_topk_candidates = int(_get(cfg, "masked_eval.topk_candidates", 150000))

    topk = int(_get(cfg, "predict.topk", 20))
    oversample = int(_get(cfg, "predict.oversample", 20))
    normalize_scores = bool(_get(cfg, "predict.normalize_scores", True))

    output_csv = str(_get(cfg, "output.csv", "inference_outputs/pred_edges.csv"))
    output_metrics_csv = _as_str_or_none(_get(cfg, "output.metrics_csv"))
    output_eval_csv = _as_str_or_none(_get(cfg, "output.eval_csv", output_metrics_csv))
    output_edges_dir = _as_str_or_none(_get(cfg, "output.edges_dir"))

    return InferConfig(
        seed=seed,
        device=device,
        mode=mode,
        cell_prefix=cell_prefix,
        cell_x_path=cell_x_path,
        cell_edge_path=cell_edge_path,
        cell_dir=cell_dir,
        cell_recursive=cell_recursive,
        cell_max_samples=cell_max_samples,
        model_name=model_name,
        model_ckpt=model_ckpt,
        num_layers=num_layers,
        hid_dim=hid_dim,
        edge_attr=edge_attr,
        model_size=model_size,
        feat_type=feat_type,
        feat_map1=feat_map1,
        feat_map2=feat_map2,
        gene_map=gene_map,
        masked_test_ratio=masked_test_ratio,
        masked_precision_at=masked_precision_at,
        masked_oversample_factor=masked_oversample_factor,
        masked_topk_candidates=masked_topk_candidates,
        topk=topk,
        oversample=oversample,
        normalize_scores=normalize_scores,
        output_csv=output_csv,
        output_metrics_csv=output_metrics_csv,
        output_eval_csv=output_eval_csv,
        output_edges_dir=output_edges_dir,
    )


def load_mapping_file(path: Path, map_location: torch.device) -> Any:
    """
    Load mapping / feature files that may be saved via `pickle.dump` or `torch.save`.

    Some pickles may contain CUDA tensors; on CPU-only runtimes this requires remapping storages to CPU.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception:
        pass

    import pickle

    original_torch_load = torch.load

    def patched_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("map_location", map_location)
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load  # type: ignore[assignment]
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    finally:
        torch.load = original_torch_load  # type: ignore[assignment]


def load_graph_from_paths(x_path: Path, edge_path: Path, use_edge_attr: bool) -> Data:
    gene_id = torch.LongTensor(np.load(x_path))
    edge_sp = sp.load_npz(edge_path).tocoo()
    edge_index = torch.tensor([edge_sp.row, edge_sp.col], dtype=torch.long)

    edge_attr = torch.FloatTensor(edge_sp.data)
    if edge_attr.numel() > 0:
        edge_attr = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min() + 1e-8)
        edge_attr = edge_attr.unsqueeze(1)
    else:
        edge_attr = edge_attr.view(-1, 1)

    return Data(
        x=gene_id.unsqueeze(1),  # id features; model.py switches to embedding for x.shape[1] == 1
        id=gene_id,
        edge_index=edge_index,
        edge_attr=edge_attr if use_edge_attr else None,
    )


@dataclass(frozen=True)
class GraphSpec:
    name: str
    prefix: str
    x_path: Path
    edge_path: Path


def resolve_prefix_from_paths(x_path: Path, edge_path: Path) -> str:
    if not x_path.name.endswith(X_SUFFIX):
        raise ValueError(f"cell.x_path must end with {X_SUFFIX!r}: {x_path}")
    if not edge_path.name.endswith(EDGE_SUFFIX):
        raise ValueError(f"cell.edge_path must end with {EDGE_SUFFIX!r}: {edge_path}")
    prefix = str(x_path)[: -len(X_SUFFIX)]
    expected_edge = prefix + EDGE_SUFFIX
    if str(edge_path) != expected_edge:
        raise ValueError(f"cell.edge_path must match cell.x_path prefix: expected {expected_edge}, got {edge_path}")
    return prefix


def discover_graphs_in_dir(root: Path, recursive: bool) -> list[GraphSpec]:
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    pattern = "**/*" if recursive else "*"
    x_files = list(root.glob(f"{pattern}{X_SUFFIX}"))
    edge_files = list(root.glob(f"{pattern}{EDGE_SUFFIX}"))

    x_map: dict[Path, Path] = {}
    for x in x_files:
        base = Path(str(x)[: -len(X_SUFFIX)])
        x_map[base] = x

    specs: list[GraphSpec] = []
    for e in edge_files:
        base = Path(str(e)[: -len(EDGE_SUFFIX)])
        x = x_map.get(base)
        if x is None:
            continue
        specs.append(GraphSpec(name=base.name, prefix=str(base), x_path=x, edge_path=e))

    specs.sort(key=lambda s: s.name)
    return specs


def infer_in_dim_from_state_dict(state: dict[str, Any], model_name: str) -> int | None:
    if model_name.startswith("vgae"):
        w = state.get("x_encoder.0.weight")
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[1])
        return None

    w = state.get("lin_layer.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[1])
    return None


def build_model(config: InferConfig, data_in_dim: int, device: torch.device) -> torch.nn.Module:
    if config.model_name.startswith("vgae") and not config.edge_attr:
        raise ValueError("For VGAE models, set `model.edge_attr: 1` in infer_config.yaml")

    ckpt_path = Path(config.model_ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (REPO_ROOT / ckpt_path).resolve()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    inferred_in_dim = infer_in_dim_from_state_dict(state, config.model_name)
    in_dim = inferred_in_dim if inferred_in_dim is not None else int(data_in_dim)

    if config.model_name in ["gcn", "gat", "gatv2", "sage", "gin"]:
        model = GNN_Backbone(in_dim=in_dim, hid_dim=config.hid_dim, num_layers=config.num_layers, gnn=config.model_name)
    elif config.model_name.startswith("vgae"):
        gnn_model = config.model_name.split("_")[1]
        model = VGAEModel(in_dim=in_dim, hid_dim=config.hid_dim, num_layers=config.num_layers, gnn=gnn_model)
    else:
        raise ValueError(f"Unknown model: {config.model_name}")

    model.load_state_dict(state)
    return model.to(device).eval()


def topk_predicted_edges(
    embeddings: torch.Tensor,
    existing_edge_index: torch.Tensor,
    topk: int,
    oversample: int,
    normalize_scores: bool,
) -> list[tuple[int, int, float]]:
    z = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    n = z.size(0)

    exist_pairs = existing_edge_index.t().tolist()
    exist_set = set((min(a, b), max(a, b)) for a, b in exist_pairs)

    sim = z @ z.t()
    sim = torch.triu(sim, diagonal=1)
    flat = sim.reshape(-1)

    k = min(max(topk * oversample, topk), flat.numel())
    scores, idx = torch.topk(flat, k=k)

    if normalize_scores and scores.numel() > 0:
        smin = scores.min()
        smax = scores.max()
        scores = (scores - smin) / (smax - smin + 1e-8)

    rows = (idx // n).tolist()
    cols = (idx % n).tolist()
    scores_list = scores.tolist()

    out: list[tuple[int, int, float]] = []
    for r, c, s in zip(rows, cols, scores_list):
        if (r, c) in exist_set:
            continue
        out.append((r, c, float(s)))
        if len(out) >= topk:
            break
    return out


def masked_eval_like_legacy(
    embeddings: torch.Tensor,
    train_pos_edge_index: torch.Tensor,
    test_pos_edge_index: torch.Tensor,
    topk_list: list[int],
    topk_candidates: int,
) -> tuple[dict[int, float], dict[int, float], list[tuple[int, int, float]]]:
    topk_list = sorted({int(x) for x in topk_list if int(x) > 0})
    if not topk_list:
        return {}, {}, []

    exist_edges_set = set(tuple(sorted(edge)) for edge in train_pos_edge_index.t().cpu().numpy())
    masked_edges_set = set(tuple(sorted(edge)) for edge in test_pos_edge_index.t().cpu().numpy())

    similarity_matrix = torch.matmul(embeddings, embeddings.t())
    similarity_matrix = similarity_matrix.triu(diagonal=1)
    node_nums = embeddings.size(0)

    k_candidates = min(int(topk_candidates), similarity_matrix.numel())
    score, indices = torch.topk(similarity_matrix.view(-1), k=k_candidates)
    row_indices = (indices // node_nums).cpu().numpy()
    col_indices = (indices % node_nums).cpu().numpy()
    confidence_scores = score.cpu().numpy()

    pred_edges = np.stack([row_indices, col_indices], axis=1).tolist()
    pred_edges_set = set(tuple(sorted(edge)) for edge in pred_edges)

    pred_edges_with_scores = [(tuple(sorted(edge)), float(s)) for edge, s in zip(pred_edges, confidence_scores)]
    pred_edges_with_scores = sorted(pred_edges_with_scores, key=lambda x: x[1], reverse=True)

    pred_edges_filtered: list[tuple[int, int]] = []
    pred_edges_filtered_with_scores: list[tuple[tuple[int, int], float]] = []
    for edge, s in pred_edges_with_scores:
        if edge not in exist_edges_set:
            pred_edges_filtered.append(edge)
            pred_edges_filtered_with_scores.append((edge, s))

    precision: dict[int, float] = {}
    for k in topk_list:
        kk = int(k)
        corr_nums = len(set(pred_edges_filtered[:kk]) & masked_edges_set)
        precision[kk] = corr_nums / kk

    ranked_filtered = [(u, v, s) for ((u, v), s) in pred_edges_filtered_with_scores]
    return precision, ranked_filtered


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_header = list(metrics.keys())

    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(new_header)
            w.writerow([metrics[k] for k in new_header])
        return

    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        try:
            existing_header = next(r)
        except StopIteration:
            existing_header = []
        existing_rows = list(r)

    if existing_header == new_header:
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([metrics[k] for k in new_header])
        return

    union_header = list(existing_header)
    for k in new_header:
        if k not in union_header:
            union_header.append(k)

    existing_index = {k: i for i, k in enumerate(existing_header)}
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(union_header)
        for row in existing_rows:
            out_row = []
            for k in union_header:
                idx = existing_index.get(k)
                out_row.append(row[idx] if idx is not None and idx < len(row) else "")
            w.writerow(out_row)
        w.writerow([metrics.get(k, "") for k in union_header])


def write_pred_edges_csv(
    path: Path,
    predicted_edges: list[tuple[int, int, float]],
    gene_ids: list[int] | None,
    reversed_gene_map: dict[int, str] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if predicted_edges:
        scores = np.asarray([s for _, _, s in predicted_edges], dtype=np.float64)
        smin = float(scores.min())
        smax = float(scores.max())
        denom = (smax - smin) + 1e-8
        predicted_edges = [(u, v, float((s - smin) / denom)) for (u, v, s) in predicted_edges]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gene1", "gene2"])
        for u, v, s in predicted_edges:
            if gene_ids is None:
                g1, g2 = str(u), str(v)
            elif reversed_gene_map is None:
                g1, g2 = str(gene_ids[u]), str(gene_ids[v])
            else:
                g1 = reversed_gene_map.get(gene_ids[u], str(gene_ids[u]))
                g2 = reversed_gene_map.get(gene_ids[v], str(gene_ids[v]))
            w.writerow([g1, g2])


def append_legacy_eval_csv(
    path: Path,
    cell_prefix: str,
    model: str,
    model_size: int,
    feat: str,
    edge_attr: int,
    precisions: dict[int, float],
    topk_list: list[int],
) -> None:
    """
    Preserve the old `evaluate_all.csv` row format (no header), one row per k:
    cell,model,model_size,feat,precision(%),topk,edge_attr
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for k in topk_list:
            p = float(precisions.get(k, 0.0)) * 100.0
            f.write(f"{cell_prefix},{model},{model_size},{feat},{p:.6f},{k},{edge_attr}\n")


def load_one_graph(
    config: InferConfig,
    spec: GraphSpec,
    device: torch.device,
    feat_map1: Any | None,
    feat_map2: Any | None,
    gene_map: Any | None,
) -> tuple[Data, list[int] | None, dict[int, str] | None]:
    reversed_gene_map: dict[int, str] | None = None
    if gene_map is not None:
        reversed_gene_map = {v: k for k, v in gene_map.items()}

    if config.feat_type == "id":
        data = load_graph_from_paths(spec.x_path, spec.edge_path, use_edge_attr=bool(config.edge_attr))
        gene_ids = data.id.detach().cpu().numpy().tolist()
        return data, gene_ids, reversed_gene_map

    class _Args:
        pass

    args = _Args()
    args.feat = config.feat_type
    args.edge_attr = int(config.edge_attr)
    args.device = device

    data = get_data(spec.prefix, args, feat_map1, feat_map2, gene_map)
    gene_ids = data.id.detach().cpu().numpy().tolist()
    return data, gene_ids, reversed_gene_map


def output_path_for_sample(config: InferConfig, sample_name: str) -> Path:
    if config.output_edges_dir:
        base = Path(config.output_edges_dir)
        if not base.is_absolute():
            base = (REPO_ROOT / base).resolve()
        return (base / f"{sample_name}.pred_edges.csv").resolve()
    out = Path(config.output_csv)
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()
    return out


def run_one_graph(
    config: InferConfig,
    spec: GraphSpec,
    device: torch.device,
    feat_map1: Any | None,
    feat_map2: Any | None,
    gene_map: Any | None,
) -> tuple[Path, dict[str, Any] | None, dict[int, float] | None, dict[int, float] | None]:
    data, gene_ids, reversed_gene_map = load_one_graph(
        config=config,
        spec=spec,
        device=device,
        feat_map1=feat_map1,
        feat_map2=feat_map2,
        gene_map=gene_map,
    )

    data.edge_index, data.edge_attr = convert_to_directed_edge(data.edge_index, data.edge_attr)

    if config.mode == "eval_masked":
        data = train_test_split_edges(data, val_ratio=0.0, test_ratio=float(config.masked_test_ratio))
    else:
        data.train_pos_edge_index = data.edge_index
        data.train_pos_edge_attr = data.edge_attr

    data = data.to(device)
    model = build_model(config, data_in_dim=int(data.x.size(1)), device=device)

    with torch.no_grad():
        if config.edge_attr:
            embeddings = model(data.x, data.train_pos_edge_index, data.train_pos_edge_attr)
        else:
            embeddings = model(data.x, data.train_pos_edge_index, None)

    out_path = output_path_for_sample(config, spec.name)

    if config.mode == "eval_masked":
        topk_list = sorted(set(int(x) for x in config.masked_precision_at))
        precisions, preds = masked_eval_like_legacy(
            embeddings=embeddings,
            train_pos_edge_index=data.train_pos_edge_index,
            test_pos_edge_index=data.test_pos_edge_index,
            topk_list=topk_list,
            topk_candidates=int(config.masked_topk_candidates),
        )
        # Per request: eval_masked only prints metrics, no prediction CSV or metrics files.
        return out_path, precisions

    preds = topk_predicted_edges(
        embeddings=embeddings,
        existing_edge_index=data.train_pos_edge_index,
        topk=int(config.topk),
        oversample=int(config.oversample),
        normalize_scores=bool(config.normalize_scores),
    )
    write_pred_edges_csv(out_path, preds, gene_ids, reversed_gene_map)
    return out_path, None, 


def resolve_single_graph_spec(config: InferConfig) -> GraphSpec:
    if config.cell_x_path and config.cell_edge_path:
        x_path = Path(config.cell_x_path)
        edge_path = Path(config.cell_edge_path)
        prefix = resolve_prefix_from_paths(x_path, edge_path)
        name = x_path.stem.replace("_x", "")
        return GraphSpec(name=name, prefix=prefix, x_path=x_path, edge_path=edge_path)

    if not config.cell_prefix:
        raise ValueError("No input specified: set cell.x_path/edge_path or cell.dir or cell.prefix")
    prefix = config.cell_prefix

    return GraphSpec(
        name=Path(prefix).name,
        prefix=prefix,
        x_path=Path(prefix + X_SUFFIX),
        edge_path=Path(prefix + EDGE_SUFFIX),
    )


def run(config: InferConfig) -> None:
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    need_feature_maps = config.feat_type != "id"
    feat_map1 = feat_map2 = gene_map = None
    if need_feature_maps:
        if config.feat_type in ["g-anno", "all"]:
            feat_map1 = load_mapping_file((REPO_ROOT / config.feat_map1).resolve(), map_location=device)
        if config.feat_type in ["s-anno", "all"]:
            feat_map2 = load_mapping_file((REPO_ROOT / config.feat_map2).resolve(), map_location=device)
    if config.gene_map:
        gene_map = load_mapping_file((REPO_ROOT / config.gene_map).resolve(), map_location=torch.device("cpu"))

    if config.cell_dir:
        specs = discover_graphs_in_dir(Path(config.cell_dir), recursive=bool(config.cell_recursive))
        if not specs:
            raise FileNotFoundError(f"No *_x.npy + *_edge.npz pairs found under: {config.cell_dir}")
        if config.cell_max_samples is not None:
            specs = specs[: max(0, int(config.cell_max_samples))]
    else:
        specs = [resolve_single_graph_spec(config)]

    metrics_csv_path = None
    if config.output_metrics_csv:
        metrics_csv_path = Path(config.output_metrics_csv)
        if not metrics_csv_path.is_absolute():
            metrics_csv_path = (REPO_ROOT / metrics_csv_path).resolve()

    eval_csv_path = None
    if config.output_eval_csv:
        eval_csv_path = Path(config.output_eval_csv)
        if not eval_csv_path.is_absolute():
            eval_csv_path = (REPO_ROOT / eval_csv_path).resolve()

    for spec in specs:
        out_path, precisions = run_one_graph(
            config=config,
            spec=spec,
            device=device,
            feat_map1=feat_map1,
            feat_map2=feat_map2,
            gene_map=gene_map,
        )
        if config.mode == "eval_masked" and precisions:
            topk_list = sorted(set(int(x) for x in config.masked_precision_at))
            for k in topk_list:
                p = float(precisions.get(k, 0.0))
                print(f"[{spec.name}] precision@{k}: {p:.6f}")
            continue

        print(f"[{config.mode}] saved {spec.name} predictions -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="YAML-configured inference (see infer_config.yaml).")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to infer_config.yaml")
    cli_args = parser.parse_args()

    config_path = Path(cli_args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    config = load_config(config_path)
    run(config)


if __name__ == "__main__":
    main()