#!/usr/bin/env python3
"""Train the GNN priority model on every instance in data/train (1080 files: .sm, .rcp, .json).

Method: REINFORCE policy gradient (the training method of the original TBGAT paper).
  For each training project, per step:
    1. The GNN scores every activity.
    2. Draw K sample orders around those scores (Gumbel noise = Plackett-Luce sampling).
    3. Decode each order with SSGS and measure its makespan.
    4. Orders that beat the sample average become more likely, worse ones less likely.
  10% of the projects are held back; after every epoch the model's plain prediction is
  decoded on them, and the epoch with the best validation gap is kept (epoch 0 = the
  untrained model, which equals the LST rule — so training can only keep or improve it).
Finally the GNN is compared with every classic rule on train, validation and the test set.

Usage: python scripts/train.py [--epochs 40] [--samples 8]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import ssgs  # noqa: E402
from src.algorithms import gnn  # noqa: E402
from src.algorithms.rules import rules  # noqa: E402
from src.parser import parse  # noqa: E402

MODELS = ROOT / "data" / "models"


def instance_files(folder):
    return sorted(f for f in (ROOT / "data" / folder).iterdir() if f.suffix in (".sm", ".rcp", ".json"))


def gap(p, priority):
    return (ssgs.decode(p, priority)[1] - p.lower_bound) / p.lower_bound


def log_prob(scores, order):
    """Plackett-Luce log-probability of picking activities in this order."""
    s = scores[order]
    return (s - torch.logcumsumexp(s.flip(0), 0).flip(0)).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(4)

    files = instance_files("train")
    projects = [parse(f) for f in files]
    print(f"Training set: {len(projects)} instances from data/train "
          f"({', '.join(f'{sum(f.suffix == e for f in files)} {e}' for e in ('.sm', '.rcp', '.json'))})")
    random.shuffle(projects)
    n_val = len(projects) // 10
    val, train = projects[:n_val], projects[n_val:]
    graphs = {id(p): gnn.graph_tensors(p) for p in projects}

    config = {"dim": 32, "heads": 4, "layers": 2}
    model = gnn.PriorityGNN(**config)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def validation_gap():
        model.eval()
        with torch.no_grad():
            return sum(gap(p, model(*graphs[id(p)]).tolist()) for p in val) / len(val)

    best = validation_gap()
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    history = [{"epoch": 0, "mean_sample_gap_%": None, "val_gap_%": round(100 * best, 3), "kept": True}]
    print(f"Epoch  0 (untrained = LST rule): validation gap {100 * best:.2f}%")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        sample_gap = 0.0
        for p in train:
            scores = model(*graphs[id(p)])
            noisy = scores.detach() - torch.log(-torch.log(torch.rand(args.samples, p.n)))   # Gumbel samples
            orders = noisy.argsort(dim=1, descending=True)
            gaps = torch.tensor([gap(p, row.tolist()) for row in noisy])
            advantage = (gaps.mean() - gaps) / (gaps.std() + 1e-6)   # positive = better than average
            loss = -sum(a * log_prob(scores, o) for a, o in zip(advantage, orders)) / args.samples
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sample_gap += gaps.mean().item()
        v = validation_gap()
        history.append({"epoch": epoch, "mean_sample_gap_%": round(100 * sample_gap / len(train), 3),
                        "val_gap_%": round(100 * v, 3)})
        kept = v < best - 1e-9
        history[-1]["kept"] = kept
        if kept:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
        print(f"Epoch {epoch:2d}: sample gap {100 * sample_gap / len(train):.2f}%  validation gap {100 * v:.2f}%"
              f"{'  (best so far, kept)' if kept else ''}  [{time.time() - t0:.0f}s]", flush=True)

    model.load_state_dict(best_state)
    torch.save({"config": config, "weights": best_state}, gnn.MODEL_PATH)
    gnn._model = None

    print("One-shot decode quality (mean gap to lower bound, lower is better)")
    report = {}
    for label, group in (("train", train), ("validation", val), ("test", [parse(f) for f in instance_files("test")])):
        g_gnn = [gap(p, gnn.predict(p)) for p in group]
        scores = {"GNN (trained)": sum(g_gnn) / len(group)}
        for rule in rules(group[0]):
            scores[rule] = sum(gap(p, rules(p)[rule]) for p in group) / len(group)
        best_rule = min((v, k) for k, v in scores.items() if k != "GNN (trained)")
        wins = sum(g <= min(gap(p, r) for r in rules(p).values()) + 1e-12 for g, p in zip(g_gnn, group))
        report[label] = {"instances": len(group), "mean_gap_%": {k: round(100 * v, 2) for k, v in scores.items()},
                         "gnn_at_least_as_good_as_every_rule": wins}
        print(f"  {label:10s} ({len(group)}): GNN {100 * scores['GNN (trained)']:.2f}%  |  best rule "
              f"{best_rule[1]} {100 * best_rule[0]:.2f}%  |  GNN >= every rule on {wins}/{len(group)}")

    (MODELS / "training_log.json").write_text(json.dumps({
        "training_instances": len(files), "method": "REINFORCE (policy gradient), Plackett-Luce sampling",
        "samples_per_step": args.samples, "model": config,
        "parameters": sum(t.numel() for t in model.parameters()),
        "epochs": history, "best_validation_gap_%": round(100 * best, 3), "evaluation": report}, indent=1))
    print(f"Saved {gnn.MODEL_PATH.relative_to(ROOT)} and data/models/training_log.json")


if __name__ == "__main__":
    main()
