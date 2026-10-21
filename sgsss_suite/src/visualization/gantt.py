"""
================================================================================
src/visualization/gantt.py — Interactive Gantt Chart Generator
================================================================================
Generates high-fidelity, interactive Plotly Gantt charts for RCPSP schedules:
  • Horizontal bar representation (Start Time → Finish Time)
  • Dynamic height scaling for projects of any size (N=10 to N=120)
  • Rich hover tooltips with duration and resource allocations
  • Clean gridlines and time unit ticks.
"""

from __future__ import annotations

import math
from typing import List, Optional

import plotly.graph_objects as go
import plotly.express as px

from src.model import Project, ScheduleEntry


# ==============================================================================
# SECTION SUMMARY: Interactive Plotly Gantt Chart Builder
# What is in this section: Builds an interactive horizontal bar Gantt timeline for
# activities, calculates dynamic figure height, injects resource demand tooltips,
# and configures professional time-grid styling.
# ==============================================================================
def create_gantt_chart(
    schedule: List[ScheduleEntry],
    project: Optional[Project] = None,
    title: str = "📅 RCPSP Activity Schedule (Gantt Timeline)",
) -> go.Figure:
    """
    Generate an interactive Plotly Gantt chart from schedule entries.

    Args:
        schedule: List of ScheduleEntry objects.
        project:  Optional Project object for resource requirement details.
        title:    Chart title.

    Returns:
        Plotly Figure object.
    """
    # --------------------------------------------------------------------------
    # Subsection: Schedule Validation & Dynamic Canvas Sizing
    # What is in this section: Computes project makespan horizon and dynamic height.
    # --------------------------------------------------------------------------
    if not schedule:
        fig = go.Figure()
        fig.update_layout(title="No Schedule Available")
        return fig

    sorted_entries = sorted(schedule, key=lambda e: e.activity_id)
    max_time = max((e.finish_time for e in sorted_entries), default=1)
    num_activities = len(sorted_entries)
    dynamic_height = max(420, num_activities * 24 + 110)

    fig = go.Figure()
    palette = px.colors.qualitative.Bold

    # --------------------------------------------------------------------------
    # Subsection: Activity Bar Construction & Resource Tooltips
    # What is in this section: Creates horizontal traces for each task with duration,
    # start/finish time labels, and multi-resource allocation details on hover.
    # --------------------------------------------------------------------------
    for idx, entry in enumerate(sorted_entries):
        duration = entry.finish_time - entry.start_time
        act_id = entry.activity_id
        label = f"Task {act_id}"
        color = palette[idx % len(palette)]

        # Resource demand hover detail
        res_text = ""
        if project and act_id in project.activities:
            reqs = project.activities[act_id].resource_requirements
            req_str = ", ".join(f"R{r}:{v}" for r, v in reqs.items() if v > 0)
            if req_str:
                res_text = f"<br>Resources: {req_str}"

        hover_info = (
            f"<b>Activity {act_id}</b><br>"
            f"Start: {entry.start_time}<br>"
            f"Finish: {entry.finish_time}<br>"
            f"Duration: {duration} time units"
            f"{res_text}"
        )

        bar_text = f"{entry.start_time}→{entry.finish_time}" if duration > 0 else "0"

        fig.add_trace(go.Bar(
            x=[duration],
            y=[label],
            base=[entry.start_time],
            orientation='h',
            name=label,
            text=bar_text,
            textposition='auto',
            hoverinfo='text',
            hovertext=hover_info,
            marker=dict(
                color=color,
                line=dict(color='rgba(0, 0, 0, 0.25)', width=1),
            ),
        ))

    # --------------------------------------------------------------------------
    # Subsection: Gridline Styling & Layout Customization
    # What is in this section: Applies clean time ticks, reversed y-axis, and white theme.
    # --------------------------------------------------------------------------
    dtick_step = max(1, math.ceil(max_time / 20))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        barmode='stack',
        showlegend=False,
        height=dynamic_height,
        margin=dict(l=90, r=40, t=50, b=50),
        xaxis=dict(
            title="Time Units (Periods / Days)",
            range=[-0.5, max_time + 1.5],
            dtick=dtick_step,
            showgrid=True,
            gridcolor='#E2E8F0',
            zeroline=True,
            zerolinecolor='#CBD5E1',
        ),
        yaxis=dict(
            title="Activity ID",
            autorange="reversed",
            showgrid=True,
            gridcolor='#F1F5F9',
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    return fig

