"""
================================================================================
src/visualization/__init__.py — Interactive Visualization Package
================================================================================
Contains Plotly chart generators:
  • gantt.py:       create_gantt_chart()
  • resources.py:   create_resource_timeline_charts()
  • convergence.py: create_convergence_chart()
"""

# ==============================================================================
# SECTION SUMMARY: Visualization Component Imports
# What is in this section: Imports interactive chart generation functions for Gantt
# schedules, resource loading timelines, and multi-solver convergence curves.
# ==============================================================================
from src.visualization.gantt import create_gantt_chart
from src.visualization.resources import create_resource_timeline_charts
from src.visualization.convergence import create_convergence_chart

# ==============================================================================
# SECTION SUMMARY: Public Module Exports
# What is in this section: Exposes the primary visualization API for UI and reporting.
# ==============================================================================
__all__ = [
    "create_gantt_chart",
    "create_resource_timeline_charts",
    "create_convergence_chart",
]

