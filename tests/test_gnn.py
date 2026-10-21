"""
Unit Tests for RCPSP Graph Neural Network (GNN) Modules.

Verifies:
1. GraphConverter accurately constructs node matrices and DAG edge tensors.
2. RCPSPDualHeadGNN forward pass produces expected shapes and bounded outputs.
3. GNNPriorityScheduler outputs 100% mathematically feasible schedules via SSGS.
4. Pretrained / trained checkpoint loading and inference reliability.
"""

import os
from pathlib import Path
import pytest
import torch

from src.evaluation import Evaluator
from src.ml.gnn_model import ALGORITHM_CLASSES, GNNPriorityScheduler, RCPSPDualHeadGNN
from src.ml.graph_converter import GraphConverter
from src.parser import Parser


TEST_INSTANCE = "data/instances/j301_1.sm"
CHECKPOINT_PATH = "models/gnn_dual_head.pt"


def test_graph_converter():
    """Verify conversion of an RCPSP instance to GraphData."""
    assert os.path.exists(TEST_INSTANCE), f"Missing test instance: {TEST_INSTANCE}"
    project = Parser.parse(TEST_INSTANCE)

    graph_data = GraphConverter.project_to_graph(project, as_torch=True)

    # 32 activities in j301_1 (30 jobs + dummy start + dummy sink)
    assert graph_data.num_nodes == 32
    assert graph_data.x.shape == (32, 12)
    assert graph_data.edge_index.shape[0] == 2
    assert graph_data.edge_index.shape[1] > 0
    assert graph_data.rev_edge_index.shape == graph_data.edge_index.shape
    assert graph_data.cpm_makespan > 0

    # Ensure node features are finite and within reasonable bounds
    assert not torch.isnan(graph_data.x).any()
    assert not torch.isinf(graph_data.x).any()
    assert (graph_data.x >= 0.0).all()


def test_gnn_forward_pass():
    """Verify dual-head forward pass of RCPSPDualHeadGNN."""
    project = Parser.parse(TEST_INSTANCE)
    graph_data = GraphConverter.project_to_graph(project, as_torch=True)

    model = RCPSPDualHeadGNN(in_features=12, hidden_dim=32, num_layers=2)
    model.eval()

    with torch.no_grad():
        node_p, algo_logits, makespan_pred = model(
            graph_data.x,
            graph_data.edge_index,
            graph_data.rev_edge_index,
        )

    # Node priority head checks
    assert node_p.shape == (graph_data.num_nodes,)
    assert (node_p >= 0.0).all() and (node_p <= 1.0).all()

    # Graph algorithm head checks
    assert algo_logits.shape == (1, len(ALGORITHM_CLASSES))


def test_gnn_priority_scheduler_feasibility():
    """Verify that GNN priority seeding produces valid schedules via SSGS."""
    project = Parser.parse(TEST_INSTANCE)
    scheduler = GNNPriorityScheduler()

    result = scheduler.solve(project)

    assert result.algorithm == "GNN-Priority"
    assert result.makespan > 0
    assert result.evaluations == 1
    assert result.runtime < 0.5  # Should easily complete in < 500 ms

    # Check 100% mathematical constraint compliance
    is_valid = Evaluator.check_constraints(project, result.schedule)
    assert is_valid is True, "GNN-Priority schedule violated precedence or resource constraints!"


def test_trained_checkpoint_inference():
    """Verify inference with saved checkpoint if available."""
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip("Model checkpoint not found. Run train_gnn.py to generate it.")

    project = Parser.parse(TEST_INSTANCE)
    scheduler = GNNPriorityScheduler(model_path=CHECKPOINT_PATH)

    result = scheduler.solve(project)

    assert result.makespan > 0
    is_valid = Evaluator.check_constraints(project, result.schedule)
    assert is_valid is True

    # Test graph algorithm head
    graph_data = GraphConverter.project_to_graph(project, as_torch=True)
    best_algo, probs = scheduler.model.predict_best_algorithm(graph_data)

    assert best_algo in ALGORITHM_CLASSES
    assert len(probs) == len(ALGORITHM_CLASSES)
    assert abs(sum(probs.values()) - 1.0) < 1e-4
