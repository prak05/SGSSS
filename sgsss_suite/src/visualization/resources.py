"""
================================================================================
src/visualization/resources.py — Resource Utilization Charts
================================================================================
Generates time-phased resource loading profiles for each renewable resource:
  • Cumulative demand over time t ∈ [0, Makespan]
  • Capacity threshold overlay (red dashed line)
  • Immediate visual audit of resource feasibility (demand ≤ capacity).
"""

from __future__ import annotations

from typing import Dict, List, Union

import plotly.graph_objects as go

from src.model import Project, ScheduleEntry


# ==============================================================================
# SECTION SUMMARY: Resource Utilization & Capacity Profile Generator
# What is in this section: Computes time-phased resource consumption profiles across
# project horizon, plots shaded demand curves, overlays capacity thresholds, and
# verifies that resource usage remains <= capacity at all time periods.
# ==============================================================================
def create_resource_timeline_charts(
    project: Project,
    schedule: List[ScheduleEntry],
) -> Dict[Union[int, str], go.Figure]:
    """
    Generate time-phased utilization charts for each project resource.

    Args:
        project:  RCPSP Project object with capacity definitions.
        schedule: List of ScheduleEntry objects.

    Returns:
        Dictionary mapping resource_id to Plotly Figure.
    """
    max_time = max((e.finish_time for e in schedule), default=0)
    figs: Dict[Union[int, str], go.Figure] = {}

    for r_id, res_obj in project.resources.items():
        # ----------------------------------------------------------------------
        # Subsection: Time-Phased Resource Demand Aggregation
        # What is in this section: Aggregates active task requirements across time.
        # ----------------------------------------------------------------------
        cap = res_obj.capacity
        usage = [0] * (max_time + 1)

        for e in schedule:
            req = project.activities[e.activity_id].resource_requirements.get(r_id, 0)
            if req > 0:
                for t in range(e.start_time, e.finish_time):
                    usage[t] += req

        peak = max(usage) if usage else 0
        fig = go.Figure()

        # ----------------------------------------------------------------------
        # Subsection: Loading Curve & Capacity Threshold Overlay
        # What is in this section: Draws blue shaded demand curve and red dashed
        # capacity upper-bound limit line.
        # ----------------------------------------------------------------------
        fig.add_trace(go.Scatter(
            x=list(range(max_time + 1)),
            y=usage,
            mode='lines',
            fill='tozeroy',
            name='Resource Loading',
            line=dict(color='#2563EB', width=2),
            hoverinfo='x+y',
        ))

        fig.add_hline(
            y=cap,
            line_dash="dash",
            line_color="#DC2626",
            line_width=2,
            annotation_text=f"Capacity Limit ({cap})",
            annotation_position="top right",
            annotation_font=dict(color="#DC2626", size=11),
        )

        # ----------------------------------------------------------------------
        # Subsection: Layout Customization
        # What is in this section: Sets titles, axes ranges, and clean white theme.
        # ----------------------------------------------------------------------
        res_title = getattr(res_obj, "name", str(r_id)) or str(r_id)
        fig.update_layout(
            title=dict(
                text=f"📊 Resource Timeline: {res_title} (Peak: {peak} / Cap: {cap})",
                font=dict(size=14),
            ),
            xaxis=dict(
                title="Time Step (t)",
                showgrid=True,
                gridcolor='#E2E8F0',
            ),
            yaxis=dict(
                title="Units Allocated",
                range=[0, max(cap, peak) * 1.15 + 1],
                showgrid=True,
                gridcolor='#E2E8F0',
            ),
            height=300,
            margin=dict(l=60, r=40, t=45, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
        )
        figs[r_id] = fig

    return figs

