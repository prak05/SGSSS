"""Plotly figures for the UI: precedence graph, Gantt chart, resource use, convergence."""
from __future__ import annotations

from typing import Dict

import plotly.graph_objects as go

from src.algorithms import PROPOSED
from src.evaluation import resource_profile
from src.model import Project, Result

# Fixed colour per algorithm (validated categorical palette; colour follows the algorithm, never its rank)
COLOURS = {"TS": "#2a78d6", "PSS": "#eb6834", "GA": "#1baf7a", "SA": "#eda100",
           "BCO": "#e87ba4", "PRH": "#008300"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, font=dict(family="Inter, sans-serif", color=INK),
                      margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fcfcfb",
                      hoverlabel=dict(font_family="Inter, sans-serif"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, color=MUTED)
    return fig


def precedence_graph(p: Project) -> go.Figure:
    """Activities placed left-to-right by earliest start; red nodes have zero slack (critical)."""
    level, rows = {}, {}
    for i in p.topo_order:
        level[i] = max((level[q] + 1 for q in p.preds[i]), default=0)
        rows.setdefault(level[i], []).append(i)
    y = {i: r - (len(col) - 1) / 2 for col in rows.values() for r, i in enumerate(col)}
    ex, ey = [], []
    for i in range(p.n):
        for s in p.succs[i]:
            ex += [level[i], level[s], None]
            ey += [y[i], y[s], None]
    critical = [p.lst[i] == p.est[i] for i in range(p.n)]
    fig = go.Figure([
        go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#bbb", width=1), hoverinfo="skip"),
        go.Scatter(x=[level[i] for i in range(p.n)], y=[y[i] for i in range(p.n)], mode="markers+text",
                   text=[str(i + 1) for i in range(p.n)], textposition="middle center",
                   marker=dict(size=22, color=["#e34948" if c else "#2a78d6" for c in critical],
                               line=dict(color="#fcfcfb", width=2)),
                   textfont=dict(color="white", size=9),
                   hovertext=[f"{p.task_names[i]}<br>duration {p.durations[i]}<br>slack {p.lst[i] - p.est[i]}"
                              for i in range(p.n)], hoverinfo="text")])
    _style(fig, 440)
    fig.update_layout(showlegend=False, title="Precedence graph — arrows run left to right; red = critical (zero slack)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def gantt(p: Project, r: Result) -> go.Figure:
    tasks = [i for i in sorted(range(p.n), key=lambda i: r.starts[i]) if p.durations[i] > 0]
    fig = go.Figure(go.Bar(
        y=[p.task_names[i] for i in tasks], x=[p.durations[i] for i in tasks],
        base=[r.starts[i] for i in tasks], orientation="h", marker_color=COLOURS.get(r.algorithm, "#2a78d6"),
        marker_line=dict(color="#fcfcfb", width=2),
        hovertext=[f"{p.task_names[i]}: {r.starts[i]} → {r.starts[i] + p.durations[i]}<br>" +
                   ", ".join(f"{n}={d}" for n, d in zip(p.resource_names, p.demands[i]) if d) for i in tasks],
        hoverinfo="text"))
    _style(fig, max(340, 20 * len(tasks)))
    fig.update_layout(title=f"{r.algorithm} schedule — makespan {r.makespan} periods", xaxis_title="time period",
                      yaxis=dict(autorange="reversed"))
    return fig


def resources(p: Project, r: Result) -> go.Figure:
    fig = go.Figure()
    for k, row in enumerate(resource_profile(p, r.starts)):
        fig.add_trace(go.Scatter(x=list(range(len(row) + 1)), y=row + [row[-1] if row else 0], line=dict(width=2),
                                 line_shape="hv", name=f"{p.resource_names[k]} (capacity {p.capacities[k]})"))
    _style(fig, 340)
    fig.update_layout(title="Resource use per period — never above capacity", xaxis_title="time period",
                      yaxis_title="units in use", hovermode="x unified")
    return fig


def convergence(results: Dict[str, Result], lower_bound: int = None) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Scatter(x=list(range(1, len(r.best_trace) + 1)), y=r.best_trace, name=name,
                                 line=dict(color=COLOURS.get(name), width=3 if name in PROPOSED else 2),
                                 line_shape="hv"))
    ends = {}                                                         # direct labels; merge equal end values
    for name, r in results.items():
        ends.setdefault(r.best_trace[-1], []).append((name, len(r.best_trace)))
    for value, group in ends.items():
        fig.add_annotation(x=max(x for _, x in group), y=value, text=" " + " · ".join(n for n, _ in group),
                           showarrow=False, xanchor="left", font=dict(color=INK, size=11))
    if lower_bound:
        fig.add_hline(y=lower_bound, line_dash="dot", line_color=MUTED,
                      annotation_text=f"lower bound {lower_bound}", annotation_font_color=MUTED)
    _style(fig, 440)
    fig.update_layout(title="Best makespan found so far, at every evaluation (lower is better)",
                      xaxis_title="evaluation number", yaxis_title="best makespan (periods)", hovermode="x unified")
    return fig


def attention_heatmap(p: Project, weights, title: str) -> go.Figure:
    """weights[i][j] = how much activity i (row) listens to activity j (column)."""
    labels = [f"{i + 1}" for i in range(p.n)]
    blues = [[0, "#fcfcfb"], [0.25, "#9ec5f4"], [0.5, "#5598e7"], [0.75, "#256abf"], [1, "#0d366b"]]
    fig = go.Figure(go.Heatmap(z=weights, x=labels, y=labels, colorscale=blues, zmin=0, zmax=1, xgap=1, ygap=1,
                               hovertemplate="activity %{y} listens to %{x}: %{z:.2f}<extra></extra>"))
    _style(fig, 430)
    fig.update_layout(title=title, xaxis_title="listens to activity", yaxis=dict(title="activity", autorange="reversed"))
    return fig


def one_shot_comparison(makespans: Dict[str, int], lower_bound: int) -> go.Figure:
    names = list(makespans)
    fig = go.Figure(go.Bar(x=names, y=[makespans[n] for n in names], text=[makespans[n] for n in names],
                           textposition="outside", textfont=dict(color=INK),
                           marker_color=["#2a78d6" if n.startswith("GNN") else "#c3c2b7" for n in names]))
    fig.add_hline(y=lower_bound, line_dash="dot", line_color=MUTED, annotation_text=f"lower bound {lower_bound}")
    _style(fig, 390)
    fig.update_layout(title="One decode, no search: the GNN's order (blue) vs. every classic rule — lower is better",
                      yaxis_title="makespan (periods)")
    return fig
