"""
================================================================================
src/visualization/convergence.py — Convergence Curves Generator
================================================================================
Compares makespan descent trajectories across solvers:
  • X-axis: SSGS Evaluation Count
  • Y-axis: Best Feasible Makespan Found
  • Distinct colors and markers for each of the 5 solvers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import plotly.graph_objects as go

from src.model import AlgorithmResult


# ==============================================================================
# SECTION SUMMARY: Color Palette Mapping
# What is in this section: Standardized brand color mappings for each of the 5 algorithms
# (GA: Blue, SA: Amber, BCO: Green, PRH: Purple, TS: Red).
# ==============================================================================
COLOR_MAP = {
    "GA": "#2563EB",   # Blue
    "SA": "#D97706",   # Amber
    "BCO": "#059669",  # Green
    "PRH": "#7C3AED",  # Purple
    "TS": "#DC2626",   # Red
}


# ==============================================================================
# SECTION SUMMARY: Solver Convergence Curves Builder
# What is in this section: Plots makespan descent trajectories across schedule
# evaluations for all 5 metaheuristic solvers using step-shaped lines and markers.
# ==============================================================================
def create_convergence_chart(
    results: Dict[str, AlgorithmResult],
    title: str = "📈 Solver Convergence Comparison (SSGS Evaluations vs Makespan)",
) -> go.Figure:
    """
    Generate an interactive Plotly comparison chart of convergence histories.

    Args:
        results: Dictionary mapping algorithm name to AlgorithmResult.
        title:   Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # --------------------------------------------------------------------------
    # Subsection: Step-Shape Trace Construction
    # What is in this section: Iterates through each algorithm's evaluation history
    # and constructs step-down curves reflecting monotonic makespan improvements.
    # --------------------------------------------------------------------------
    for name, res in results.items():
        hist: List[Tuple[int, int]] = res.convergence_history
        if not hist:
            continue

        xs = [pt[0] for pt in hist]
        ys = [pt[1] for pt in hist]

        # Extend line to total evaluations if final point was earlier
        if xs and res.evaluations > xs[-1]:
            xs.append(res.evaluations)
            ys.append(ys[-1])

        color = COLOR_MAP.get(name.upper(), "#475569")

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines+markers',
            name=f"{name} (Best: {res.makespan}, Time: {res.runtime:.2f}s)",
            line=dict(color=color, width=2.5, shape='hv'),  # Step-like descent
            marker=dict(size=6, symbol='circle'),
            hoverinfo='name+x+y',
        ))

    # --------------------------------------------------------------------------
    # Subsection: Multi-Solver Legend & Layout Configuration
    # What is in this section: Configures horizontal top legend, axes labels, and white styling.
    # --------------------------------------------------------------------------
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(
            title="SSGS Schedule Evaluations",
            showgrid=True,
            gridcolor='#E2E8F0',
        ),
        yaxis=dict(
            title="Best Makespan (Days / Periods)",
            showgrid=True,
            gridcolor='#E2E8F0',
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
        ),
        height=420,
        margin=dict(l=60, r=40, t=70, b=45),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    return fig

