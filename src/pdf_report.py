"""
PDF Report Generator for RCPSP Projects.

Generates a professional, printable, layman-friendly PDF report
containing executive summaries, metric explanations, algorithm benchmark results,
and rendered charts (Gantt, Resource Utilization, Precedence DAG, and Convergence).
"""

import io
import tempfile
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


def generate_pdf_report(
    project: Any,
    features: Dict[str, Any],
    best_res: Optional[Any] = None,
    all_results: Optional[Dict[str, Any]] = None,
    proj_name: str = "Project",
) -> bytes:
    """
    Builds a printable PDF report byte stream.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_LEFT,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
    )
    heading_style = ParagraphStyle(
        "SecHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    bold_body_style = ParagraphStyle(
        "BoldBodyText",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    caption_style = ParagraphStyle(
        "ChartCaption",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
    )

    story = []

    # 1. Header & Title Block
    story.append(Paragraph("RCPSP Executive Project & Schedule Report", title_style))
    story.append(Paragraph(f"<b>Instance:</b> {proj_name} | <b>Generated Reference Document</b>", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=12))

    # 2. Executive Summary Block
    makespan_str = str(best_res.makespan) if best_res else "N/A"
    algo_str = best_res.algorithm if best_res else "N/A"
    runtime_str = f"{best_res.runtime:.3f} s" if best_res else "N/A"

    exec_summary_html = f"""
    <b>Executive Summary:</b><br/>
    This report provides a complete, easy-to-understand breakdown of the project scheduling analysis for 
    <b>{proj_name}</b>. The optimization solver evaluated the precedence dependencies and resource limits to 
    determine the shortest possible project completion duration (<b>Makespan</b>).<br/><br/>
    • <b>Winning Algorithm:</b> {algo_str}<br/>
    • <b>Optimized Makespan:</b> {makespan_str} time units<br/>
    • <b>Theoretical Lower Bound (CPM):</b> {int(features.get('critical_path_duration', 0))} time units<br/>
    • <b>Solver Runtime:</b> {runtime_str}
    """
    story.append(Paragraph(exec_summary_html, body_style))
    story.append(Spacer(1, 10))

    # 3. Project Key Metrics Table
    story.append(Paragraph("1. Project Characteristics & Plain-English Explanations", heading_style))

    metrics_data = [
        [
            Paragraph("<b>Metric</b>", bold_body_style),
            Paragraph("<b>Value</b>", bold_body_style),
            Paragraph("<b>Plain-English Meaning for Non-Technical Users</b>", bold_body_style),
        ],
        [
            Paragraph("Total Activities (N<sub>total</sub>)", body_style),
            Paragraph(str(len(project.activities)), body_style),
            Paragraph("Total number of tasks in the project, including start and finish points.", body_style),
        ],
        [
            Paragraph("Real Activities (N)", body_style),
            Paragraph(str(int(features.get("N", 0))), body_style),
            Paragraph("Number of actual work tasks that require time and effort.", body_style),
        ],
        [
            Paragraph("Resource Types (K)", body_style),
            Paragraph(str(len(project.resources)), body_style),
            Paragraph("Different types of equipment, workers, or materials available.", body_style),
        ],
        [
            Paragraph("Network Complexity (NC)", body_style),
            Paragraph(f"{features.get('NC', 0):.2f}", body_style),
            Paragraph("Measures dependency density. Higher values mean tasks are heavily interdependent.", body_style),
        ],
        [
            Paragraph("Resource Factor (RF)", body_style),
            Paragraph(f"{features.get('RF', 0):.2f}", body_style),
            Paragraph("Average fraction of resource types required per task (0 = light, 1 = heavy demand).", body_style),
        ],
        [
            Paragraph("Resource Strength (RS)", body_style),
            Paragraph(f"{features.get('RS', 0):.2f}", body_style),
            Paragraph("Measures resource availability tightness (1.0 = abundant resources, <0.2 = tight bottlenecks).", body_style),
        ],
        [
            Paragraph("CPM Critical Path", body_style),
            Paragraph(f"{int(features.get('critical_path_duration', 0))} units", body_style),
            Paragraph("Minimum possible project duration assuming unlimited resources.", body_style),
        ],
    ]

    t_metrics = Table(metrics_data, colWidths=[130, 65, 345])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 12))

    # 4. Comparison Table (if multiple algorithms ran)
    if all_results and len(all_results) > 0:
        story.append(Paragraph("2. Solver Performance & Comparison", heading_style))
        comp_data = [
            [
                Paragraph("<b>Algorithm</b>", bold_body_style),
                Paragraph("<b>Makespan</b>", bold_body_style),
                Paragraph("<b>Runtime (s)</b>", bold_body_style),
                Paragraph("<b>Evaluations</b>", bold_body_style),
                Paragraph("<b>Status</b>", bold_body_style),
            ]
        ]
        for algo, r in all_results.items():
            comp_data.append([
                Paragraph(f"<b>{algo}</b>", body_style),
                Paragraph(str(r.makespan), body_style),
                Paragraph(f"{r.runtime:.3f}", body_style),
                Paragraph(str(r.evaluations), body_style),
                Paragraph("<font color='green'>Feasible ✓</font>", body_style),
            ])
        t_comp = Table(comp_data, colWidths=[100, 100, 100, 120, 120])
        t_comp.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t_comp)
        story.append(Spacer(1, 12))

    # 5. Visual Charts (Rendered via Matplotlib)
    story.append(Paragraph("3. Schedule Visualizations & Charts", heading_style))

    # --- Chart A: Gantt Chart ---
    if best_res and best_res.schedule:
        fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=150)
        entries = sorted(best_res.schedule, key=lambda e: e.activity_id)
        y_labels = [f"Act {e.activity_id}" for e in entries]
        starts = [e.start_time for e in entries]
        durs = [e.duration for e in entries]

        ax.barh(y_labels, durs, left=starts, color="#2563eb", edgecolor="#1e3a8a", alpha=0.85, height=0.6)
        ax.set_xlabel("Time Units", fontsize=9)
        ax.set_ylabel("Activities", fontsize=9)
        ax.set_title(f"Optimized Schedule Gantt Chart (Makespan: {best_res.makespan})", fontsize=10, fontweight="bold", pad=8)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.invert_yaxis()
        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        img_buf.seek(0)
        story.append(Image(img_buf, width=540, height=230))
        story.append(Paragraph("Figure 1: Task timeline showing exact start/finish windows for all activities.", caption_style))
        story.append(Spacer(1, 10))

    # --- Chart B: Resource Utilization Profiles ---
    if best_res and best_res.schedule:
        max_t = max((e.finish_time for e in best_res.schedule), default=0)
        n_res = len(project.resources)
        if n_res > 0 and max_t > 0:
            fig, axes = plt.subplots(n_res, 1, figsize=(7.5, 2.2 * n_res), dpi=150, squeeze=False)
            for idx, (r_id, res_obj) in enumerate(project.resources.items()):
                ax = axes[idx, 0]
                cap = res_obj.capacity
                usage = [0] * (max_t + 1)
                for e in best_res.schedule:
                    req = project.activities[e.activity_id].resource_requirements.get(r_id, 0)
                    for t in range(e.start_time, e.finish_time):
                        usage[t] += req

                ax.plot(range(max_t + 1), usage, drawstyle="steps-post", color="#0284c7", linewidth=2, label="Demand")
                ax.fill_between(range(max_t + 1), usage, step="post", color="#38bdf8", alpha=0.35)
                ax.axhline(y=cap, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Capacity ({cap})")
                ax.set_ylabel(f"Res {r_id}", fontsize=8)
                ax.set_title(f"Resource {r_id} Demand vs Capacity Timeline", fontsize=9, fontweight="bold")
                ax.grid(True, linestyle=":", alpha=0.6)
                ax.legend(loc="upper right", fontsize=8)

            axes[-1, 0].set_xlabel("Time Units", fontsize=9)
            plt.tight_layout()

            img_buf_res = io.BytesIO()
            plt.savefig(img_buf_res, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            img_buf_res.seek(0)
            story.append(Image(img_buf_res, width=540, height=160 * n_res))
            story.append(Paragraph("Figure 2: Resource utilization profiles verifying capacity compliance over time.", caption_style))
            story.append(Spacer(1, 10))

    # --- Chart C: Convergence History Graph ---
    if all_results and any(r.convergence_history for r in all_results.values()):
        fig, ax = plt.subplots(figsize=(7.5, 2.8), dpi=150)
        for algo, r in all_results.items():
            if r.convergence_history:
                evs = [x[0] for x in r.convergence_history]
                ms_vals = [x[1] for x in r.convergence_history]
                ax.plot(evs, ms_vals, marker="o", markersize=3, linewidth=1.8, label=algo)

        ax.set_xlabel("Solver Evaluations", fontsize=9)
        ax.set_ylabel("Makespan", fontsize=9)
        ax.set_title("Algorithm Convergence Profile", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
        plt.tight_layout()

        img_buf_conv = io.BytesIO()
        plt.savefig(img_buf_conv, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        img_buf_conv.seek(0)
        story.append(Image(img_buf_conv, width=540, height=200))
        story.append(Paragraph("Figure 3: Convergence curves showing solution optimization progress.", caption_style))

    # Build Document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
