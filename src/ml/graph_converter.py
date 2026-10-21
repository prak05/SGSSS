"""
Graph Conversion Module for AI-RCPSP.

Converts an RCPSP Project instance (Activities, Precedence Relations, Resources)
into a structured Graph representation suitable for Graph Neural Networks (GNNs).

Features per node (Activity):
1. Normalized Duration: p_i / total_duration
2. Mean Resource Demand Ratio: mean(r_{i,k} / R_k)
3. Peak Resource Demand Ratio: max(r_{i,k} / R_k)
4. Earliest Start Time (EST) normalized by CPM makespan
5. Earliest Finish Time (EFT) normalized by CPM makespan
6. Latest Start Time (LST) normalized by CPM makespan
7. Latest Finish Time (LFT) normalized by CPM makespan
8. Total Slack: (LST - EST) normalized by CPM makespan
9. In-degree (Predecessor count) normalized by N
10. Out-degree (Successor count) normalized by N
11. Is Dummy Start indicator (1.0 or 0.0)
12. Is Dummy Finish indicator (1.0 or 0.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.features import compute_cpm_schedule
from src.model import Project

# Try importing torch; if not yet installed, provide numpy fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


@dataclass
class GraphData:
    """
    Container for graph representation of an RCPSP instance.
    """
    x: Union["torch.Tensor", np.ndarray]
    edge_index: Union["torch.Tensor", np.ndarray]
    activity_ids: List[int]
    activity_to_idx: Dict[int, int]
    num_nodes: int
    num_edges: int
    cpm_makespan: int
    feature_names: List[str] = field(default_factory=list)
    rev_edge_index: Optional[Union["torch.Tensor", np.ndarray]] = None

    def to_torch(self, device: Optional[str] = None):
        """Ensure tensors are torch.Tensor and moved to device."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for to_torch(). Please install torch.")
        
        tx = self.x if isinstance(self.x, torch.Tensor) else torch.tensor(self.x, dtype=torch.float32)
        tedge = self.edge_index if isinstance(self.edge_index, torch.Tensor) else torch.tensor(self.edge_index, dtype=torch.long)
        trev = None
        if self.rev_edge_index is not None:
            trev = self.rev_edge_index if isinstance(self.rev_edge_index, torch.Tensor) else torch.tensor(self.rev_edge_index, dtype=torch.long)

        if device is not None:
            tx = tx.to(device)
            tedge = tedge.to(device)
            if trev is not None:
                trev = trev.to(device)

        return GraphData(
            x=tx,
            edge_index=tedge,
            activity_ids=self.activity_ids,
            activity_to_idx=self.activity_to_idx,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            cpm_makespan=self.cpm_makespan,
            feature_names=self.feature_names,
            rev_edge_index=trev,
        )


NODE_FEATURE_NAMES = [
    "norm_duration",
    "mean_resource_ratio",
    "max_resource_ratio",
    "est_norm",
    "eft_norm",
    "lst_norm",
    "lft_norm",
    "slack_norm",
    "in_degree_norm",
    "out_degree_norm",
    "is_dummy_start",
    "is_dummy_finish",
]


class GraphConverter:
    """
    Converts RCPSP Project instances to Graph tensors.
    """

    @staticmethod
    def project_to_graph(
        project: Project,
        as_torch: bool = True,
        device: Optional[str] = None,
    ) -> GraphData:
        """
        Convert a Project object to GraphData.

        Parameters
        ----------
        project : Project
            The parsed RCPSP project instance.
        as_torch : bool
            If True and torch is available, returns PyTorch Tensors.
        device : Optional[str]
            Device ('cpu', 'cuda', etc.) for torch tensors.

        Returns
        -------
        GraphData
        """
        sorted_ids = sorted(project.activities.keys())
        N = len(sorted_ids)
        if N == 0:
            raise ValueError("Cannot convert an empty project with no activities.")

        activity_to_idx = {act_id: idx for idx, act_id in enumerate(sorted_ids)}

        # 1. Topological order & CPM passes
        # Simple topological sort (Kahn's algorithm)
        in_deg = {a_id: len(project.activities[a_id].predecessors) for a_id in sorted_ids}
        queue = [a_id for a_id in sorted_ids if in_deg[a_id] == 0]
        topo_order: List[int] = []

        while queue:
            curr = queue.pop(0)
            topo_order.append(curr)
            for succ in project.activities[curr].successors:
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)

        if len(topo_order) != N:
            # Fallback to sorted_ids if graph has irregularities
            topo_order = list(sorted_ids)

        es, ef, ls, lf, cpm_makespan = compute_cpm_schedule(project, topo_order)
        cpm_norm = max(1, cpm_makespan)

        total_duration = max(1, sum(act.duration for act in project.activities.values()))
        max_duration = max(1, max(act.duration for act in project.activities.values()))

        # 2. Build Node Features
        node_features = np.zeros((N, len(NODE_FEATURE_NAMES)), dtype=np.float32)

        start_node_id = min(sorted_ids)
        finish_node_id = max(sorted_ids)

        for idx, a_id in enumerate(sorted_ids):
            act = project.activities[a_id]
            dur = act.duration

            # Normalized duration (relative to max single activity duration)
            norm_dur = dur / float(max_duration)

            # Resource demand ratios
            req_ratios: List[float] = []
            for r_id, res in project.resources.items():
                req = act.resource_requirements.get(r_id, 0)
                cap = max(1, res.capacity)
                req_ratios.append(req / float(cap))

            if req_ratios:
                mean_res = float(np.mean(req_ratios))
                max_res = float(np.max(req_ratios))
            else:
                mean_res = 0.0
                max_res = 0.0

            # CPM bounds
            est_val = es.get(a_id, 0) / float(cpm_norm)
            eft_val = ef.get(a_id, dur) / float(cpm_norm)
            lst_val = ls.get(a_id, 0) / float(cpm_norm)
            lft_val = lf.get(a_id, dur) / float(cpm_norm)
            slack_val = max(0.0, (ls.get(a_id, 0) - es.get(a_id, 0)) / float(cpm_norm))

            # Degree centrality
            in_deg_val = len(act.predecessors) / float(max(1, N - 1))
            out_deg_val = len(act.successors) / float(max(1, N - 1))

            # Dummy indicators
            is_start = 1.0 if (a_id == start_node_id or (len(act.predecessors) == 0 and dur == 0)) else 0.0
            is_finish = 1.0 if (a_id == finish_node_id or (len(act.successors) == 0 and dur == 0)) else 0.0

            node_features[idx, 0] = norm_dur
            node_features[idx, 1] = mean_res
            node_features[idx, 2] = max_res
            node_features[idx, 3] = est_val
            node_features[idx, 4] = eft_val
            node_features[idx, 5] = lst_val
            node_features[idx, 6] = lft_val
            node_features[idx, 7] = slack_val
            node_features[idx, 8] = in_deg_val
            node_features[idx, 9] = out_deg_val
            node_features[idx, 10] = is_start
            node_features[idx, 11] = is_finish

        # 3. Build Edge Indices (u -> v where u is predecessor of v)
        sources: List[int] = []
        targets: List[int] = []

        for a_id in sorted_ids:
            act = project.activities[a_id]
            v_idx = activity_to_idx[a_id]
            for p_id in act.predecessors:
                if p_id in activity_to_idx:
                    u_idx = activity_to_idx[p_id]
                    sources.append(u_idx)
                    targets.append(v_idx)

        # If project has no edges, add self loops to prevent empty edge tensors
        if not sources:
            for idx in range(N):
                sources.append(idx)
                targets.append(idx)

        edge_index = np.array([sources, targets], dtype=np.int64)
        rev_edge_index = np.array([targets, sources], dtype=np.int64)
        num_edges = edge_index.shape[1]

        data = GraphData(
            x=node_features,
            edge_index=edge_index,
            activity_ids=sorted_ids,
            activity_to_idx=activity_to_idx,
            num_nodes=N,
            num_edges=num_edges,
            cpm_makespan=cpm_makespan,
            feature_names=list(NODE_FEATURE_NAMES),
            rev_edge_index=rev_edge_index,
        )

        if as_torch and TORCH_AVAILABLE:
            return data.to_torch(device=device)

        return data
