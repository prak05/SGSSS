"""One-click PDF report: project summary, ranking table, how the winner won, convergence plot."""
from __future__ import annotations

import io
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

from src.benchmark import explain  # noqa: E402
from src.charts import COLOURS  # noqa: E402
from src.model import Project, Result  # noqa: E402


def _convergence_png(results: Dict[str, Result], lb: int) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=130)
    for name, r in results.items():
        ax.step(range(1, len(r.best_trace) + 1), r.best_trace, where="post", label=name, color=COLOURS.get(name))
    ax.axhline(lb, ls=":", c="k", lw=1, label=f"lower bound {lb}")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("best makespan")
    ax.legend(ncol=7, fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def pdf_report(p: Project, results: Dict[str, Result]) -> bytes:
    e = explain(p, results)
    st = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=30)
    cols = ["rank", "algorithm", "name", "makespan", "gap_to_lower_bound_%", "total_finish", "levelling",
            "best_found_at_eval", "evaluations", "runtime_s", "feasible"]
    table = Table([cols] + [[str(row[c]) for c in cols] for row in e["table"]], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                               ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#fde2e2"))]))
    story = [
        Paragraph(f"AI-RCPSP report — {p.name}", st["Title"]),
        Paragraph(f"{p.real_tasks} activities, {len(p.capacities)} resources "
                  f"({', '.join(f'{n}={c}' for n, c in zip(p.resource_names, p.capacities))}). "
                  f"Critical path {p.cpm_bound}, lower bound {p.lower_bound}. "
                  f"All solvers had the same budget of {results[e['winner']].evaluations} schedule evaluations.", st["Normal"]),
        Spacer(1, 8), Paragraph(f"Winner: {e['winner']} — makespan {e['makespan']}", st["Heading2"]),
        Paragraph(e["how"], st["Normal"]),
        Paragraph("Ranking order: makespan, then total finish time, then resource levelling, then evaluations "
                  "needed, then runtime. Every schedule was re-checked by an independent audit.", st["Normal"]),
        Spacer(1, 8), table, Spacer(1, 8),
        Image(_convergence_png(results, p.lower_bound), width=760, height=300),
    ]
    doc.build(story)
    return buf.getvalue()
