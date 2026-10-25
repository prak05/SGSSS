"""Bidirectional Graph Attention Network (GNN) that predicts activity priorities.

The project is a graph: activities are nodes, precedence arcs are edges. Each activity
starts with 10 features (rules.FEATURES). Every layer then lets each activity look in
two directions with attention:
  * upstream   — its predecessors (what must happen before it: readiness),
  * downstream — its successors   (what is waiting for it: bottleneck pressure).
Attention learns which neighbours matter most. Two layers = information from two
steps away; a mean over all nodes adds whole-project context. A final small network
turns each activity's embedding into a correction that is added to the classic LST rule
(latest start time first). Untrained, the model is exactly LST; training learns when
to deviate from it.

Training (scripts/train.py) is reinforcement learning (REINFORCE, as in the original
TBGAT paper) on 570 training projects: sample priority orders around the model's scores,
decode them with SSGS, and make shorter-makespan orders more likely.
At run time TS and PSS start from the GNN's prediction instead of a blind guess.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from src.algorithms.rules import FEATURES, features, rules
from src.model import Project

MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "gnn.pt"


class DirectedAttention(nn.Module):
    """Multi-head attention restricted to graph neighbours in one direction (plus self)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads, self.d = heads, dim // heads
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        q, k, v = (f(h).view(n, self.heads, self.d).transpose(0, 1) for f in (self.q, self.k, self.v))
        scores = (q @ k.transpose(1, 2)) / self.d ** 0.5            # heads x n x n
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        self.last = weights.detach().mean(dim=0)                    # kept for inspect(): n x n
        return (weights @ v).transpose(0, 1).reshape(n, -1)


class BiGATLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.up, self.down = DirectedAttention(dim, heads), DirectedAttention(dim, heads)
        self.mix = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, pred_mask, succ_mask):
        both = torch.cat([self.up(h, pred_mask), self.down(h, succ_mask)], dim=-1)
        return self.norm(h + self.mix(both))                        # residual keeps own features


class PriorityGNN(nn.Module):
    def __init__(self, dim: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(len(FEATURES), dim)
        self.layers = nn.ModuleList(BiGATLayer(dim, heads) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.head[-1].weight)                         # untrained = exactly the LST rule
        nn.init.zeros_(self.head[-1].bias)
        self.rule_weight = nn.Parameter(torch.tensor(20.0))

    def forward(self, x, pred_mask, succ_mask, parts=False):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, pred_mask, succ_mask)
        context = h.mean(dim=0, keepdim=True).expand_as(h)            # whole-project view
        correction = self.head(torch.cat([h, context], dim=-1)).squeeze(-1)
        rule = self.rule_weight * -x[:, 1]
        return (rule, correction) if parts else rule + correction    # LST rule + learned correction


def graph_tensors(p: Project):
    """Feature matrix and the two neighbour masks (each node also attends to itself)."""
    eye = np.eye(p.n, dtype=bool)
    pred, succ = eye.copy(), eye.copy()
    for i, ps in enumerate(p.preds):
        pred[i, ps] = True
    for i, ss in enumerate(p.succs):
        succ[i, ss] = True
    return (torch.tensor(features(p), dtype=torch.float32),
            torch.from_numpy(pred), torch.from_numpy(succ))


_model: Optional[PriorityGNN] = None


def load_model() -> Optional[PriorityGNN]:
    global _model
    if _model is None and MODEL_PATH.exists():
        saved = torch.load(MODEL_PATH, map_location="cpu")
        _model = PriorityGNN(**saved["config"])
        _model.load_state_dict(saved["weights"])
        _model.eval()
    return _model


def predict(p: Project) -> List[float]:
    """Priority vector from the trained GNN (falls back to the LFT rule if not trained yet)."""
    model = load_model()
    if model is None:
        return rules(p)["LFT (latest finish time)"]
    with torch.no_grad():
        return model(*graph_tensors(p)).tolist()


def inspect(p: Project) -> Optional[dict]:
    """Everything the GNN computes for one project, for the dashboard's 'GNN inside' tab."""
    model = load_model()
    if model is None:
        return None
    x, pred_mask, succ_mask = graph_tensors(p)
    with torch.no_grad():
        rule, correction = model(x, pred_mask, succ_mask, parts=True)
    score = rule + correction
    rank = {int(i): r + 1 for r, i in enumerate(torch.argsort(score, descending=True))}
    lst_rank = {int(i): r + 1 for r, i in enumerate(torch.argsort(rule, descending=True))}
    return {
        "features": x.numpy(), "feature_names": FEATURES,
        "attention": {f"layer {li + 1} {side}": getattr(layer, attr).last.numpy()
                      for li, layer in enumerate(model.layers)
                      for side, attr in (("upstream (predecessors)", "up"), ("downstream (successors)", "down"))},
        "lst_term": rule.numpy(), "correction": correction.numpy(), "score": score.numpy(),
        "rank": [rank[i] for i in range(p.n)], "lst_rank": [lst_rank[i] for i in range(p.n)],
        "parameters": sum(t.numel() for t in model.parameters()),
    }
