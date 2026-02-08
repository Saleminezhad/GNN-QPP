import argparse
import glob
import logging
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ==================== Argument Parser ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Train or inference GNN regressor on query-query graphs")
    parser.add_argument("--graphs_dir", required=True, help="Directory containing training graph_*.pt files")
    parser.add_argument("--test_dirs", nargs="*", help="Optional directories for evaluation")
    parser.add_argument("--pred_out", default="predictions", help="Output directory for predictions and model")
    parser.add_argument("--model", choices=["graphsage", "gcn", "gat"], default="gcn", help="GNN architecture (GCN uses edge weights, SAGEConv doesn't)")
    parser.add_argument("--gat_heads", type=int, default=4, help="Number of attention heads for GAT (only used when --model gat)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of GNN layers")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of GNN layers (1-2 is usually enough for star graphs)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Fraction of graphs for training")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of graphs for validation")
    parser.add_argument("--clip_grad", type=float, default=2.0, help="Gradient clipping norm")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train", action="store_true", help="If set, run in training mode. Otherwise, inference only.")
    parser.add_argument("--reg_layers", type=int, choices=[1, 2, 3], default=2, help="Number of layers in the final regressor MLP")
    parser.add_argument("--use_edge_weights", action="store_true", default=True, help="Use edge weights (query similarity) in GCN convolutions")
    parser.add_argument("--log_embeddings", action="store_true", help="(Optional) Log sample query embeddings and neighbor influences")
    parser.add_argument("--effective_topk", type=int, default=None, help="If set, limit each graph to top-k neighbors at load time")
    parser.add_argument("--feature_mode", choices=["emb_only", "emb_plus_perf", "emb_plus_neighbor_stats"], default="emb_plus_perf", help="How to construct node features at runtime")
    parser.add_argument("--label_metric", choices=["ndcg", "map"], default="ndcg", help="Which metric to use as regression target during training/eval")
    parser.add_argument("--train_metric", choices=["ndcg", "map"], default="ndcg", help="Which metric to treat as 'perf' inside node features")

    return parser.parse_args()


# ==================== Dataset ====================

class QueryGraphDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for query–query graphs with:
      • effective_topk: dynamically limit #neighbors at load time
      • feature_mode: emb_only / emb_plus_perf / emb_plus_neighbor_stats
      • train_metric: which metric to use as 'perf' (ndcg or map)

    Raw graphs on disk contain:
      - x: [num_nodes, feature_dim]
        * now: [embedding, ndcg, map] for each node
      - edge_index: [2, num_edges]
      - edge_weight: [num_edges] (similarity scores)
      - y: [2] = [ndcg, map] for the center query
      - metadata: { "qid": ... }
      - (optional) query_mask: bool mask for query node (node 0)

    """
    def __init__(
        self,
        graph_paths: List[str],
        feature_mode: str = "emb_plus_perf",
        effective_topk: int | None = None,
        train_metric: str = "ndcg",
    ):
        assert train_metric in {"ndcg", "map"}, "train_metric must be 'ndcg' or 'map'"
        self.graph_paths = graph_paths
        self.feature_mode = feature_mode
        self.effective_topk = effective_topk
        self.train_metric = train_metric

        logging.info(
            f"Loaded dataset with {len(graph_paths)} graphs | "
            f"feature_mode={feature_mode} | effective_topk={effective_topk} | "
            f"train_metric={train_metric}"
        )

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx: int) -> Data:
        graph_path = self.graph_paths[idx]
        try:
            data: Data = torch.load(graph_path, weights_only=False)
        except (EOFError, RuntimeError) as e:
            raise RuntimeError(
                f"Failed to load graph file {graph_path}. "
                "The file is likely empty or corrupted; regenerate the dataset."
            ) from e

        # ---------- 1) Ensure core fields exist ----------
        # query_mask: mark node 0 as query
        if not hasattr(data, "query_mask"):
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[0] = True
            data.query_mask = mask

        # edge_weight must exist — otherwise raise an error
        if not hasattr(data, "edge_weight") or data.edge_weight is None:
            raise ValueError(
                f"Graph {self.graph_paths[idx]} has no edge_weight field. "
                "All graphs must include edge weights."
            )

        # qid from metadata or filename
        if hasattr(data, "metadata") and isinstance(data.metadata, dict) and "qid" in data.metadata:
            data.qid = data.metadata["qid"]
        else:
            data.qid = os.path.splitext(
                os.path.basename(self.graph_paths[idx])
            )[0].replace("graph_", "")

        # ---------- 2) Apply effective_topk truncation (query + first k neighbors) ----------
        if self.effective_topk is not None:
            max_nodes = 1 + self.effective_topk
            max_nodes = min(max_nodes, data.num_nodes)

            node_mask = torch.arange(data.num_nodes) < max_nodes

            data.x = data.x[node_mask]
            data.query_mask = data.query_mask[node_mask]

            ei = data.edge_index
            edge_mask = (ei[0] < max_nodes) & (ei[1] < max_nodes)
            data.edge_index = ei[:, edge_mask]

            if hasattr(data, "edge_weight") and data.edge_weight is not None:
                data.edge_weight = data.edge_weight[edge_mask]

        # ---------- 3) Apply feature transform using chosen train_metric ----------
        data = apply_feature_mode(data, self.feature_mode, self.train_metric)

        return data


def apply_feature_mode(data: Data, mode: str, train_metric: str) -> Data:
    """
    Modify data.x based on the chosen feature mode and train_metric.

    Raw data.x is assumed to be:
        x_raw = [embedding, ndcg, map]

    We first pick:
        perf = ndcg  if train_metric == "ndcg"
            = map   if train_metric == "map"

    Then we reuse the logic built for:
        x_feat = [embedding, perf]
    """
    x_raw = data.x

    # x_raw: [N, D_raw], D_raw = emb_dim + 2
    emb       = x_raw[:, :-2]     # [N, emb_dim]
    ndcg_feat = x_raw[:, -2:-1]   # [N, 1]
    map_feat  = x_raw[:, -1:]     # [N, 1]

    if train_metric == "ndcg":
        perf = ndcg_feat
    elif train_metric == "map":
        perf = map_feat
    else:
        raise ValueError(f"Unknown train_metric: {train_metric}")

    x_feat = torch.cat([emb, perf], dim=-1)   # [N, emb_dim+1]

    if mode == "emb_only":
        data.x = emb

    elif mode == "emb_plus_perf":
        data.x = x_feat


    elif mode == "emb_plus_neighbor_stats":
        # neighbors = nodes 1..N-1
        if x_feat.size(0) > 1:
            neighbor_perf = perf[1:]  # [N-1, 1]
            # ✅ keep 2D shape [1,1]
            mean_perf = neighbor_perf.mean(dim=0, keepdim=True)  # [1,1]
            max_perf  = neighbor_perf.max(dim=0, keepdim=True).values  # [1,1]
        else:
            mean_perf = torch.zeros((1, 1), dtype=x_feat.dtype, device=x_feat.device)
            max_perf  = torch.zeros((1, 1), dtype=x_feat.dtype, device=x_feat.device)

        # Query node feature: [emb_q, mean_perf, max_perf]  -> [1, emb_dim+2]
        query_feat = torch.cat([emb[0:1], mean_perf, max_perf], dim=-1)

        # Neighbor nodes: just embeddings, padded to same dim
        neighbor_feat = emb[1:]  # [N-1, emb_dim]

        target_dim = query_feat.size(1)
        pad = target_dim - neighbor_feat.size(1)
        if pad > 0:
            neighbor_feat = torch.nn.functional.pad(neighbor_feat, (0, pad))

        data.x = torch.cat([query_feat, neighbor_feat], dim=0)  # [N, emb_dim+2]

    else:
        raise ValueError(f"Unknown feature_mode: {mode}")

    return data


def filter_valid_graphs(paths: List[str]) -> List[str]:
    """
    Drop empty / unreadable graph files to avoid dataloader crashes.
    Only a size check is done here to keep it cheap; torch.load will still
    throw with a clear message if a non-empty file is corrupted.
    """
    valid = []
    skipped = []

    for p in paths:
        try:
            if os.path.getsize(p) <= 0:
                skipped.append(p)
                continue
        except OSError:
            skipped.append(p)
            continue
        valid.append(p)

    if skipped:
        logging.warning(
            f"Skipping {len(skipped)} empty/corrupt graph files. "
            f"Using {len(valid)} remaining graphs."
        )
    return valid


# ==================== Model ====================

class QueryRegressor(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers,
        model_type="gcn",
        dropout=0.1,
        reg_layers=2,
        gat_heads=4,          # <-- new
    ):
        super().__init__()
        self.dropout = dropout
        self.model_type = model_type
        self.num_layers = num_layers
        self.gat_heads = gat_heads

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_dim if i == 0 else hidden_dim
            out_channels = hidden_dim

            if model_type == "gcn":
                self.convs.append(GCNConv(in_channels, out_channels))
            elif model_type == "graphsage":
                self.convs.append(SAGEConv(in_channels, out_channels))
            elif model_type == "gat":
                # concat=False keeps output dim = out_channels
                self.convs.append(
                    GATConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        heads=gat_heads,
                        concat=False,
                        dropout=dropout,
                        add_self_loops=False,
                    )
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        
        # regressor stays exactly the same
        if reg_layers == 1:
            self.regressor = nn.Linear(hidden_dim, out_dim)
        elif reg_layers == 2:
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        elif reg_layers == 3:
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            raise ValueError(f"reg_layers must be 1–3, got {reg_layers}")
        

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        
        for conv in self.convs:
            if self.model_type == "gcn" and edge_weight is not None:
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        query_emb = x[data.query_mask]  # [batch_size, hidden_dim]
        pred = self.regressor(query_emb)  # [batch_size, 1]
        return pred.view(-1)  # [batch_size]

    @torch.no_grad()
    def get_query_embedding(self, data: Data) -> torch.Tensor:
        self.eval()
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        
        for conv in self.convs:
            if self.model_type == "gcn" and edge_weight is not None:
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
        
        query_emb = x[data.query_mask]
        return query_emb.squeeze(0).cpu()


# ==================== Training & Evaluation ====================
def _reshape_y(y: torch.Tensor) -> torch.Tensor:
    """
    Ensure y is [B, 2] = [ndcg, map].

    When batching, PyG may produce:
      - shape [2]   for a single graph
      - shape [B*2] for B graphs (1D)
      - shape [B, 2] already

    This helper converts it to [B, 2].
    """
    if y.dim() == 2:
        return y
    if y.dim() == 1:
        if y.numel() % 2 != 0:
            raise ValueError(f"Cannot reshape y of length {y.numel()} to [N, 2]")
        return y.view(-1, 2)
    if y.dim() == 0:
        # degenerate case, not expected but we guard anyway
        return y.view(1, 1)
    raise ValueError(f"Unexpected y.dim()={y.dim()}, shape={tuple(y.shape)}")

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    label_metric: str,
    clip_grad: float | None = None
) -> float:
    """
    Train for one epoch.

    label_metric: "ndcg" or "map" – which dimension of y to use.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    label_idx = 0 if label_metric == "ndcg" else 1
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        preds = model(batch)  # [B]

        # 🔧 make sure y is [B, 2] = [ndcg, map]
        y = _reshape_y(batch.y)
        target = y[:, label_idx].to(device)  # [B]
        
        loss = F.mse_loss(preds, target)
        loss.backward()
        
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs
    
    avg_loss = total_loss / num_samples
    return avg_loss


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    label_metric: str
) -> Tuple[float, float, float]:
    """
    Evaluate on validation/test set using given label_metric ("ndcg" or "map").
    """
    model.eval()
    preds_all = []
    labels_all = []
    total_mse = 0.0
    num_samples = 0

    label_idx = 0 if label_metric == "ndcg" else 1
    
    for batch in loader:
        batch = batch.to(device)
        
        preds = model(batch)  # [B]

        # 🔧 reshape y
        y = _reshape_y(batch.y)
        target = y[:, label_idx].to(device)  # [B]
        
        preds_all.append(preds.cpu())
        labels_all.append(target.cpu())
        
        total_mse += F.mse_loss(preds, target, reduction="sum").item()
        num_samples += batch.num_graphs
    
    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    
    mse = total_mse / num_samples
    mae = F.l1_loss(preds_all, labels_all).item()
    
    rho = float("nan")
    if spearmanr is not None and len(preds_all) > 1:
        rho, _ = spearmanr(preds_all.numpy(), labels_all.numpy())
    
    return mse, mae, rho


def split_paths(
    paths: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    rng.shuffle(paths)
    
    n_total = len(paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_paths = paths[:n_train]
    val_paths = paths[n_train:n_train + n_val]
    test_paths = paths[n_train + n_val:]
    
    return train_paths, val_paths, test_paths


def log_diagnostics(model, sample_graph, args):
    logging.info("\n" + "=" * 70)
    logging.info("MODEL & DATA DIAGNOSTICS")
    logging.info("=" * 70)
    
    logging.info(f"✓ Node feature dimension: {sample_graph.num_node_features}")
    
    if hasattr(sample_graph, "edge_weight") and sample_graph.edge_weight is not None:
        ew = sample_graph.edge_weight.numpy()
        logging.info(f"✓ Edge weights present: shape {ew.shape}")
        logging.info(f"  Range: [{ew.min():.4f}, {ew.max():.4f}]")
        logging.info(f"  Mean: {ew.mean():.4f}, Std: {ew.std():.4f}")
    else:
        logging.warning("⚠️  No edge weights found! GCN will use uniform weights.")
    
    if hasattr(sample_graph, "query_mask"):
        logging.info(f"✓ Query mask: {sample_graph.query_mask.sum()} node(s) marked as query")
    else:
        logging.warning("⚠️  No query_mask found! (Dataset class will add it at load time.)")
    
    logging.info(f"✓ Nodes: {sample_graph.num_nodes}")
    logging.info(f"✓ Edges: {sample_graph.edge_index.shape[1]}")

    if hasattr(sample_graph, "y"):
        y = sample_graph.y
        if y.ndim == 0 or y.numel() == 1:
            logging.info(f"✓ Target (y): {y.item():.4f}")
        elif y.numel() == 2:
            logging.info(f"✓ Target (y): ndcg={y[0].item():.4f}, map={y[1].item():.4f}")
        else:
            logging.info(f"✓ Target (y): shape={tuple(y.shape)}, values={y.cpu().numpy()}")
    else:
        logging.warning("⚠️  No target y found in sample_graph")

    logging.info("\n✓ Model architecture:")
    logging.info(f"  GNN type: {args.model.upper()}")
    logging.info(f"  GNN layers: {args.num_layers}")
    logging.info(f"  Hidden dim: {args.hidden_dim}")
    logging.info(f"  Regressor layers: {args.reg_layers}")
    logging.info(f"  Dropout: {args.dropout}")
    logging.info(f"  Use edge weights: {args.use_edge_weights and args.model == 'gcn'}")
    
    logging.info("=" * 70 + "\n")


# ==================== Main ====================

def main():
    args = parse_args()
    
    # ==================== Seed & Device ====================
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    
    
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device(args.device)
    logging.info(f"Device: {device}")
    
    # ==================== Load Graphs ====================
    graph_paths = sorted(glob.glob(os.path.join(args.graphs_dir, "graph_*.pt")))
    if not graph_paths:
        raise FileNotFoundError(f"❌ No graphs found in {args.graphs_dir}")

    graph_paths = filter_valid_graphs(graph_paths)
    if not graph_paths:
        raise FileNotFoundError(f"❌ No valid graphs found in {args.graphs_dir} (all empty/corrupt)")

    logging.info(f"Found {len(graph_paths)} graph files")
    
    # ==================== Load Sample Graph & Infer Dimensions ====================
    sample_graph = torch.load(graph_paths[0], weights_only=False)
    sample_graph = apply_feature_mode(sample_graph, args.feature_mode, args.train_metric)
    in_dim = sample_graph.num_node_features
    logging.info(f"✓ Detected in_dim (node feature dimension) = {in_dim}")
    
    # ==================== Create Output Directory ====================
    os.makedirs(args.pred_out, exist_ok=True)
    model_path = os.path.join(args.pred_out, "trained_model.pt")
    
    # ==================== Initialize Model ====================
    model = QueryRegressor(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=1,
        num_layers=args.num_layers,
        model_type=args.model,
        dropout=args.dropout,
        reg_layers=args.reg_layers,
        gat_heads=args.gat_heads,   # <-- new
    ).to(device)
    
    log_diagnostics(model, sample_graph, args)
    
    # ==================== Optimizer ====================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # ==================== Data Split & DataLoaders ====================
    train_paths, val_paths, _ = split_paths(
        graph_paths,
        args.train_ratio,
        args.val_ratio,
        args.seed
    )
    
    logging.info(
        f"Train/Val/Test split: "
        f"{len(train_paths)}/{len(val_paths)}/{len(graph_paths)-len(train_paths)-len(val_paths)}"
    )
    
    train_dataset = QueryGraphDataset(
        train_paths,
        feature_mode=args.feature_mode,
        effective_topk=args.effective_topk,
        train_metric=args.train_metric,
    )
    val_dataset = QueryGraphDataset(
        val_paths,
        feature_mode=args.feature_mode,
        effective_topk=args.effective_topk,
        train_metric=args.train_metric,
    ) if val_paths else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    ) if val_dataset else None
    
    # ==================== TRAINING MODE ====================
    if args.train:
        logging.info("\n" + "=" * 70)
        logging.info("🚀 TRAINING MODE")
        logging.info("=" * 70)
        
        best_val_mse = float("inf")
        best_state = None
        patience_counter = 0
        patience = 7
        
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                label_metric=args.label_metric,
                clip_grad=args.clip_grad,
            )
            
            if val_loader:
                val_mse, val_mae, val_rho = evaluate(
                    model,
                    val_loader,
                    device,
                    label_metric=args.label_metric,
                )
                
                log_msg = (
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train MSE: {train_loss:.6f} | "
                    f"Val MSE: {val_mse:.6f} | "
                    f"MAE: {val_mae:.6f} | "
                    f"Spearman ρ: {val_rho:.4f}"
                )
                logging.info(log_msg)
                
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(
                            f"Early stopping at epoch {epoch} "
                            f"(no improvement for {patience} epochs)"
                        )
                        break
            else:
                logging.info(
                    f"Epoch {epoch:3d}/{args.epochs} | Train MSE: {train_loss:.6f}"
                )
        
        if best_state is not None:
            torch.save(best_state, model_path)
            logging.info(
                f"✅ Saved best model (Val MSE: {best_val_mse:.6f}) to {model_path}"
            )
        else:
            torch.save(model.state_dict(), model_path)
            logging.info(f"✅ Saved final model to {model_path}")
    
    # ==================== INFERENCE MODE ====================
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ No trained model found at {model_path}. Run with --train first."
            )
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"✅ Loaded pretrained model from {model_path}")
    
    # ==================== PREDICTION ON TEST SETS ====================
    if args.test_dirs:
        logging.info("\n" + "=" * 70)
        logging.info("📊 RUNNING PREDICTIONS ON TEST SETS")
        logging.info("=" * 70)
        
        model.eval()

        # Which index is the main label and which is the secondary?
        # y[:, 0] = ndcg, y[:, 1] = map
        label_idx = 0 if args.label_metric == "ndcg" else 1
        other_idx = 1 - label_idx
        
        for test_dir in args.test_dirs:
            test_graphs = sorted(glob.glob(os.path.join(test_dir, "graph_*.pt")))
            test_graphs = filter_valid_graphs(test_graphs)
            logging.info(
                f"\nEvaluating on test dir: {test_dir} with {len(test_graphs)} graphs"
            )
            if not test_graphs:
                logging.warning(f"No graphs found in test dir {test_dir}")
                continue
            
            # ✅ Use same dataset configuration as for train/val
            dataset = QueryGraphDataset(
                test_graphs,
                feature_mode=args.feature_mode,
                effective_topk=args.effective_topk,
                train_metric=args.train_metric,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
            
            rows = []
            test_mse_total = 0.0
            test_mae_total = 0.0
            num_test_samples = 0
            
            for batch in loader:
                batch = batch.to(device)
                preds = model(batch).detach().cpu()   # [B]
                
                # ✅ Make sure y is [B, 2] = [ndcg, map]
                y = _reshape_y(batch.y)

                # main target = the one we train/evaluate on
                target = y[:, label_idx].cpu()
                # secondary target = the other metric (just for record)
                target1 = y[:, other_idx].cpu()
                
                # qids: PyG will usually collate strings into list/tuple
                qids = getattr(batch, "qid", ["NA"] * len(preds))
                
                # ✅ Metrics computed ONLY on the main label
                test_mse_total += F.mse_loss(preds, target, reduction="sum").item()
                test_mae_total += F.l1_loss(preds, target, reduction="sum").item()
                num_test_samples += len(preds)
                
                # Save both metrics for logging
                for qid, p, t, t1 in zip(qids, preds, target, target1):
                    rows.append((qid, float(p), float(t), float(t1)))
            
            # Write predictions
            out_name = os.path.basename(os.path.normpath(test_dir))
            out_path = os.path.join(args.pred_out, f"{out_name}_pred.tsv")
            
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("qid\tpred\tlabel\tlabel1\n")
                for qid, p, t, t1 in rows:
                    f.write(f"{qid}\t{p:.6f}\t{t:.6f}\t{t1:.6f}\n")
            
            # Compute test metrics on the main label
            test_mse = test_mse_total / num_test_samples
            test_mae = test_mae_total / num_test_samples
            test_rho = float("nan")
            
            if spearmanr is not None and len(rows) > 1:
                preds_np = np.array([p for _, p, _, _ in rows])
                labels_np = np.array([t for _, _, t, _ in rows])  # main label only
                test_rho, _ = spearmanr(preds_np, labels_np)
            
            logging.info(f"\n[{out_name}]")
            logging.info(f"  Queries: {len(rows)}")
            logging.info(f"  MSE: {test_mse:.6f}")
            logging.info(f"  MAE: {test_mae:.6f}")
            logging.info(f"  Spearman ρ ({args.label_metric}): {test_rho:.4f}")
            logging.info(f"  Predictions saved → {out_path}")
    
    logging.info("\n✅ Done!")


if __name__ == "__main__":
    main()
