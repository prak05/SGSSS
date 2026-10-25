"""AI-RCPSP dashboard (team project).  Run:  streamlit run ui.py

Pick a project, run all six algorithms with the same budget, watch every iteration, look inside
the trained GNN, and read — from the measured data — who won and why. All explanatory text
lives in explanation.py.
"""
import csv
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

import explanation as ex
from src import charts, ssgs
from src.algorithms import BASELINES, FULL_NAMES, PROPOSED, SOLVERS, gnn
from src.algorithms.rules import rules
from src.benchmark import CRITERIA, auto_budget, auto_seed, explain, resource_pressure, run_all
from src.evaluation import audit
from src.model import Result
from src.parser import parse
from src.report import pdf_report

ROOT = Path(__file__).resolve().parent
SOURCES = {"Use cases — 66 realistic projects": ROOT / "data" / "use_cases",
           "Test set — 260 held-out benchmarks": ROOT / "data" / "test",
           "Training set — 1,080 benchmarks": ROOT / "data" / "train",
           "Small demos — 8 hand-written projects": ROOT / "data" / "use_cases_small"}

st.set_page_config(page_title="AI-RCPSP · Neural-guided Project Scheduling", page_icon="🗓️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root { --ink:#0f1b2d; --ink2:#2b3a4f; --muted:#5b6474; --line:#e3e1da; --paper:#f7f6f2; --card:#ffffff;
        --blue:#2a78d6; --orange:#eb6834; }
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family:'Inter',sans-serif; }
h1, h2, h3, h4, h5 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.01em; color:var(--ink); }
.stApp { background-color:var(--paper);
         background-image:linear-gradient(#efede7 1px, transparent 1px), linear-gradient(90deg, #efede7 1px, transparent 1px);
         background-size:32px 32px; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.2rem; max-width:1420px; }
section[data-testid="stSidebar"] { background:#fbfaf7; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .brand { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.2rem; color:var(--ink); }
.mono { font-family:'JetBrains Mono',monospace; }

/* hero — blueprint panel */
.hero { position:relative; overflow:hidden; background:var(--ink); color:#e8edf5; border-radius:20px;
        padding:30px 34px 22px 34px; margin-bottom:20px;
        background-image:linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
                         linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px);
        background-size:24px 24px; }
.hero .eyebrow { font-family:'JetBrains Mono',monospace; font-size:.74rem; letter-spacing:.18em; color:#8fb4e6; }
.hero h1 { font-family:'Space Grotesk',sans-serif !important; color:#fff; font-size:2.35rem; font-weight:700;
           margin:6px 0 8px 0; line-height:1.1; max-width:760px; }
.hero h1 em { font-style:normal; color:#8fb4e6; }
.hero p { color:#c4cfdf; max-width:700px; font-size:1.0rem; margin:0; }
.hero svg { position:absolute; right:34px; top:30px; opacity:.95; }
@media (max-width: 1250px) { .hero svg { display:none; } }
.hero .stats { display:flex; flex-wrap:wrap; gap:0; margin-top:22px; border-top:1px solid rgba(255,255,255,.12); padding-top:14px; }
.hero .stat { padding:0 22px 0 0; margin-right:22px; border-right:1px solid rgba(255,255,255,.14); }
.hero .stat:last-child { border-right:none; }
.hero .stat b { display:block; font-family:'JetBrains Mono',monospace; font-size:1.25rem; color:#fff; }
.hero .stat span { font-size:.76rem; color:#9fb0c8; }

/* KPI cards */
.kpi { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px 16px 12px 16px;
       height:100%; box-shadow:0 1px 0 rgba(15,27,45,.04); }
.kpi .l { font-family:'JetBrains Mono',monospace; color:var(--muted); font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; }
.kpi .v { font-family:'JetBrains Mono',monospace; font-size:1.75rem; font-weight:600; color:var(--ink); line-height:1.2; margin-top:2px; }
.kpi .h { color:var(--muted); font-size:.76rem; margin-top:2px; }

/* tabs as a segmented control */
.stTabs [role="tablist"] { gap:4px; background:var(--card); border:1px solid var(--line); border-radius:14px;
                           padding:5px; flex-wrap:wrap; box-shadow:0 1px 0 rgba(15,27,45,.04); }
.stTabs [role="tab"] { border-radius:10px; padding:9px 18px !important; margin:0 !important; height:auto;
                       border-bottom:none !important; box-shadow:none !important; }
.stTabs [role="tab"]::after, .stTabs [role="tab"]::before { display:none !important; }
.stTabs [role="tab"] p { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:.95rem; color:var(--ink2); }
.stTabs [role="tab"]:hover { background:#f1efe9; }
.stTabs [role="tab"][aria-selected="true"] { background:var(--ink) !important; }
.stTabs [role="tab"][aria-selected="true"] p { color:#fff !important; }
.stTabs [role="tablist"] + div, .stTabs [role="tablist"] > div:not([role="tab"]) { display:none; }

/* page lead + guide */
.lead { display:flex; gap:16px; align-items:baseline; margin:14px 0 10px 0; }
.lead .no { font-family:'JetBrains Mono',monospace; font-size:.72rem; letter-spacing:.14em; color:var(--blue); white-space:nowrap; }
.lead .tx { color:var(--ink2); font-size:.98rem; }
[data-testid="stExpander"] details { background:var(--card); border:1px solid var(--line) !important; border-radius:14px !important; }
[data-testid="stExpander"] summary p { font-weight:600; color:var(--ink); }
.terms { display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:10px; margin:6px 0 16px 0; }
.term { background:var(--paper); border:1px solid var(--line); border-radius:12px; padding:10px 12px; }
.term b { font-family:'Space Grotesk',sans-serif; color:var(--ink); display:block; margin-bottom:2px; }
.term span { color:var(--ink2); font-size:.86rem; }
.fx-title { font-family:'Space Grotesk',sans-serif; font-weight:700; color:var(--ink); font-size:1.02rem; }
.fx-k { font-family:'JetBrains Mono',monospace; font-size:.66rem; letter-spacing:.14em; color:var(--blue); text-transform:uppercase; }
.fx-why { border-left:3px solid var(--orange); padding-left:10px; }
.fx-what { border-left:3px solid var(--blue); padding-left:10px; }
.sec { font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:.16em; color:var(--muted); margin:6px 0 8px 0; }

/* win tiles */
.tile { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:12px 14px; min-height:182px; }
.tile.win { border:2px solid var(--ink); box-shadow:0 6px 18px rgba(15,27,45,.12); }
.tile .name { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.08rem; color:var(--ink); }
.tile .full { color:var(--muted); font-size:.74rem; margin-bottom:8px; }
.tile .num { font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:600; color:var(--ink); }
.tile .cap { color:var(--muted); font-size:.72rem; }
.tile .bar { height:6px; border-radius:3px; background:#ecebe5; margin:5px 0 8px 0; }
.tile .bar > div { height:6px; border-radius:3px; }
.sw { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; }
.badge { display:inline-block; background:var(--ink); color:#fff; font-family:'JetBrains Mono',monospace; font-size:.6rem;
         letter-spacing:.1em; padding:2px 8px; border-radius:999px; margin-bottom:6px; }
.badge.off { visibility:hidden; }
.tag { font-family:'JetBrains Mono',monospace; font-size:.58rem; letter-spacing:.08em; padding:1px 6px; border-radius:999px;
       border:1px solid #c9c6bc; color:var(--muted); margin-left:4px; vertical-align:middle; }

/* winner card */
.winner { background:var(--card); border:1px solid var(--line); border-left:6px solid var(--ink); border-radius:16px;
          padding:20px 24px; margin-bottom:14px; }
.winner .t { font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:.16em; color:var(--blue); text-transform:uppercase; }
.winner .w { font-family:'Space Grotesk',sans-serif; font-size:2.0rem; font-weight:700; color:var(--ink); margin:2px 0 6px 0; }
.winner .s { color:var(--ink2); }
.foot { margin-top:40px; padding:16px 4px; border-top:1px solid var(--line); color:var(--muted); font-size:.82rem;
        font-family:'JetBrains Mono',monospace; letter-spacing:.04em; }
</style>""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load(path: str):
    return parse(path)


@st.cache_data(show_spinner=False)
def benchmark_wins():
    """Wins per algorithm over all unseen projects in our full evaluation (test set + use cases)."""
    wins, total = Counter(), 0
    for f in (ROOT / "docs" / "results.csv", ROOT / "docs" / "use_cases.csv"):
        if f.exists():
            for row in csv.DictReader(open(f)):
                wins[row["winner"]] += 1
                total += 1
    return wins, total


PAGE_NO = {"overview": "01", "iterations": "02", "schedules": "03", "results": "04", "gnn": "05", "evidence": "06"}


def guide(key: str):
    """Page lead + this page's own guide: the words used here and every formula, with why it is used."""
    st.markdown(f'<div class="lead"><span class="no">PAGE {PAGE_NO[key]}</span><span class="tx">{ex.TABS[key]}</span>'
                '</div>', unsafe_allow_html=True)
    page = ex.HELP.get(key)
    if not page:
        return
    with st.expander(f"📘  Guide to this page — {len(page['terms'])} key terms · {len(page['formulas'])} formulas "
                     "explained (what they mean and why we use them)"):
        st.markdown('<div class="sec">KEY TERMS ON THIS PAGE</div><div class="terms">' +
                    "".join(f'<div class="term"><b>{t}</b><span>{ex.TERMS[t]}</span></div>' for t in page["terms"]) +
                    '</div><div class="sec">FORMULAS USED ON THIS PAGE</div>', unsafe_allow_html=True)
        for i, (name, latex, means, why) in enumerate(page["formulas"], 1):
            with st.container(border=True):
                st.markdown(f'<span class="fx-k">Formula {i}</span><div class="fx-title">{name}</div>',
                            unsafe_allow_html=True)
                st.latex(latex)
                c1, c2 = st.columns(2)
                c1.markdown(f'<div class="fx-what"><span class="fx-k">What it means</span><br>{means}</div>',
                            unsafe_allow_html=True)
                c2.markdown(f'<div class="fx-why"><span class="fx-k" style="color:#c2501f">Why we use it</span><br>{why}'
                            '</div>', unsafe_allow_html=True)


def kpi(col, label, value, hint=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="h">{hint}</div></div>', unsafe_allow_html=True)


def legend(columns, extra=None):
    items = list(extra or []) + [(c, ex.explain_column(c)) for c in columns if ex.explain_column(c)]
    if items:
        with st.expander("ℹ️  What do these columns mean?"):
            st.markdown("\n".join(f"- **{c}** — {t}" for c, t in items))


def table(df: pd.DataFrame, height=None, explain=True):
    """A dataframe whose every column header has a tooltip, plus a legend below it."""
    config = {c: st.column_config.Column(help=ex.explain_column(c)) for c in df.columns if ex.explain_column(c)}
    kwargs = {"height": height} if height else {}
    st.dataframe(df, column_config=config, hide_index=True, width="stretch", **kwargs)
    if explain:
        legend(df.columns)


# ── header ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <svg width="230" height="110" viewBox="0 0 230 110" aria-hidden="true">
    <line x1="0" y1="106" x2="230" y2="106" stroke="rgba(255,255,255,.25)"/><rect x="0" y="0" width="60" height="10" rx="3" fill="#2a78d6"/><rect x="40" y="18" width="80" height="10" rx="3" fill="#eb6834"/><rect x="70" y="36" width="50" height="10" rx="3" fill="#1baf7a"/><rect x="110" y="54" width="70" height="10" rx="3" fill="#eda100"/><rect x="30" y="72" width="55" height="10" rx="3" fill="#e87ba4"/><rect x="150" y="90" width="60" height="10" rx="3" fill="#2a78d6"/>
  </svg>
  <div class="eyebrow">B.TECH FINAL-YEAR PROJECT · RIET · BUILT BY THE AI-RCPSP TEAM</div>
  <h1>Scheduling projects with a <em>neural head start</em></h1>
  <p>A graph neural network gives Tabu Search and Particle Swarm Search a smart first guess; both are then compared,
     under identical rules, with four classic methods on real benchmark projects.</p>
  <div class="stats">
    <div class="stat"><b>6</b><span>algorithms compared</span></div>
    <div class="stat"><b>1,080</b><span>projects used to train the GNN</span></div>
    <div class="stat"><b>326</b><span>unseen projects tested</span></div>
    <div class="stat"><b>100%</b><span>schedules independently verified</span></div>
  </div>
</div>""", unsafe_allow_html=True)

# ── sidebar ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div class="brand">⚙️ Control panel</div>', unsafe_allow_html=True)
st.sidebar.caption("1 · choose a project  →  2 · choose settings  →  3 · press Run")
source = st.sidebar.radio("Project source", list(SOURCES) + ["Upload your own file"])
if source == "Upload your own file":
    up = st.sidebar.file_uploader("RCPSP file (.sm, .rcp or .json)", type=["sm", "rcp", "json"])
    if not up:
        st.info("Upload a .sm, .rcp or .json project file in the sidebar to begin.")
        st.stop()
    tmp = Path(tempfile.gettempdir()) / f"rcpsp_upload_{up.name}"
    tmp.write_bytes(up.getvalue())
    path = str(tmp)
else:
    files = sorted(f.name for f in SOURCES[source].iterdir() if f.suffix in (".sm", ".rcp", ".json"))
    fmt = st.sidebar.multiselect("File formats", [".sm", ".rcp", ".json"], default=[".sm", ".rcp", ".json"])
    files = [f for f in files if Path(f).suffix in fmt]
    if not files:
        st.warning("No files match the selected formats.")
        st.stop()
    path = str(SOURCES[source] / st.sidebar.selectbox(f"Project file ({len(files)} available)", files))
try:
    p = load(path)
except Exception as exc:                                  # bad upload: say why, don't crash
    st.error(f"Could not read this file: {exc}")
    st.stop()

mode = st.sidebar.radio("Mode", ["Run all algorithms", "Single algorithm"],
                        help="Run all = the fair comparison that decides a winner.")
chosen = list(SOLVERS) if mode == "Run all algorithms" else [
    st.sidebar.selectbox("Algorithm", list(SOLVERS), format_func=lambda a: f"{a} — {FULL_NAMES[a]}")]
auto_b, auto_s, pressure = auto_budget(p), auto_seed(p), resource_pressure(p)
st.sidebar.markdown(f"**Evaluation budget:** {auto_b} per algorithm  \n**Random seed:** {auto_s}")
st.sidebar.caption(f"Set automatically from this project: {ex.AUTO_SETTINGS.format(n=p.real_tasks, pressure=pressure)}")
if st.sidebar.toggle("Set budget and seed manually", value=False):
    budget = st.sidebar.number_input("Evaluations per algorithm", 50, 20000, auto_b, step=50,
                                     help=ex.TERMS["Evaluation budget"])
    seed = st.sidebar.number_input("Random seed", 0, 10**6, auto_s, help=ex.TERMS["Random seed"])
else:
    budget, seed = auto_b, auto_s
run_key = (path, tuple(chosen), budget, seed)
if st.sidebar.button("▶  Run", type="primary", width="stretch"):
    bar = st.sidebar.progress(0.0, "starting")

    def tick(name, r):
        done = list(SOLVERS).index(name) + 1 if len(chosen) > 1 else 1
        bar.progress(done / len(chosen), f"{name} finished — makespan {r.makespan}")

    st.session_state["run"] = (run_key, run_all(p, budget, seed, names=chosen, on_done=tick))
results = st.session_state.get("run", (None, None))[1] if st.session_state.get("run", (None,))[0] == run_key else None
st.sidebar.markdown("---")
st.sidebar.caption("AI-RCPSP · team project · RIET")

# ── project KPIs ─────────────────────────────────────────────────────────────────
st.markdown(f"#### Project: `{p.name}`")
cols = st.columns(5)
kpi(cols[0], "Activities", p.real_tasks, "tasks to schedule")
res_text = ", ".join(f"{n} {c}" for n, c in zip(p.resource_names, p.capacities))
kpi(cols[1], "Resources", len(p.capacities), res_text if len(res_text) <= 48 else res_text[:46] + "…")
kpi(cols[2], "Critical path", p.cpm_bound, "shortest possible with unlimited resources")
kpi(cols[3], "Lower bound", p.lower_bound, "no schedule can be shorter")
kpi(cols[4], "Budget / algorithm", budget, f"{'auto' if (budget, seed) == (auto_b, auto_s) else 'manual'} · "
                                          f"seed {seed} · same for all")
st.write("")

tabs = st.tabs(["Overview", "Iterations", "Schedules", "Results", "GNN inside", "Benchmark evidence"])
NEED_RUN = "Press **▶ Run** in the sidebar to start."

# ── 1 Overview ───────────────────────────────────────────────────────────────────
with tabs[0]:
    guide("overview")
    st.plotly_chart(charts.precedence_graph(p), width="stretch")
    table(pd.DataFrame({
        "Activity": p.task_names, "Duration": p.durations,
        "Must follow": [", ".join(p.task_names[q] for q in ps) for ps in p.preds],
        **{f"{r} (cap {c})": [d[k] for d in p.demands] for k, (r, c) in enumerate(zip(p.resource_names, p.capacities))},
        "Earliest start": p.est, "Slack": [l - e for l, e in zip(p.lst, p.est)]}), height=380)
    st.markdown("##### The six algorithms")
    table(pd.DataFrame([{"Algorithm": a, "Name": n, "Type": t, "How it works": d}
                        for a, (n, t, d) in ex.ALGORITHMS.items()]), explain=False)

# ── 2 Iterations ─────────────────────────────────────────────────────────────────
with tabs[1]:
    guide("iterations")
    if results is None:
        st.info(NEED_RUN)
    else:
        winner = explain(p, results)["winner"] if len(results) > 1 else None
        wins, total = benchmark_wins()
        steps = max(len(r.best_trace) for r in results.values())
        led = Counter()
        for t in range(steps):
            now = {a: r.best_trace[min(t, len(r.best_trace) - 1)] for a, r in results.items()}
            led.update(a for a, v in now.items() if v == min(now.values()))
        st.markdown("##### Win schematic")
        for col, a in zip(st.columns(len(SOLVERS)), SOLVERS):
            share = wins[a] / total if total else 0
            col.markdown(f"""
<div class="tile {'win' if a == winner else ''}">
  <div class="badge {'' if a == winner else 'off'}">🏆 WINNER</div>
  <div class="name"><span class="sw" style="background:{charts.COLOURS[a]}"></span>{a}
    <span class="tag">{'PROPOSED' if a in PROPOSED else 'BASELINE'}</span></div>
  <div class="full">{FULL_NAMES[a]}</div>
  <div class="num">{wins[a]}</div><div class="cap">benchmark wins of {total}</div>
  <div class="bar"><div style="width:{100 * share:.1f}%; background:{charts.COLOURS[a]}"></div></div>
  <div class="cap">{f"Led this run at <b>{led[a]}</b> of {steps} evaluations" if a in results else "not run"}</div>
</div>""", unsafe_allow_html=True)
        st.caption(ex.TILES)

        st.plotly_chart(charts.convergence(results, p.lower_bound), width="stretch")
        st.markdown("##### Iteration table — one row per evaluation, every algorithm")
        frame = pd.DataFrame({f"{a} {kind}": pd.Series(getattr(r, field), index=range(1, len(r.trace) + 1))
                              for a, r in results.items() for kind, field in (("best", "best_trace"), ("tried", "trace"))})
        frame.index.name = "Evaluation"
        table(frame.reset_index(), height=420, explain=False)
        legend([], [("Evaluation", ex.COLUMNS["Evaluation"]), ("<algorithm> best", ex.COLUMNS["best"]),
                    ("<algorithm> tried", ex.COLUMNS["tried"]),
                    ("empty cells", "The algorithm stopped early because its schedule was provably optimal.")])
        st.download_button("⬇  Download all iterations (CSV)", frame.to_csv(), f"{p.name}_iterations.csv")
        st.markdown("##### When each algorithm improved")
        for col, (a, r) in zip(st.columns(len(results)), results.items()):
            col.markdown(f'<span class="sw" style="background:{charts.COLOURS[a]}"></span>**{a}**', unsafe_allow_html=True)
            improved = pd.DataFrame([(t + 1, b) for t, b in enumerate(r.best_trace) if t == 0 or b < r.best_trace[t - 1]],
                                    columns=["At evaluation", "New best"])
            col.dataframe(improved, hide_index=True, height=240, column_config={
                c: st.column_config.Column(help=ex.COLUMNS[c]) for c in improved.columns})
        legend(["At evaluation", "New best"])

# ── 3 Schedules ──────────────────────────────────────────────────────────────────
with tabs[2]:
    guide("schedules")
    if results is None:
        st.info(NEED_RUN)
    else:
        a = st.selectbox("Show the schedule of", list(results),
                         format_func=lambda x: f"{x} — {FULL_NAMES[x]} (makespan {results[x].makespan})")
        st.plotly_chart(charts.gantt(p, results[a]), width="stretch")
        st.caption("Each bar is one activity, from its start (left edge) to its finish (right edge). "
                   "Hover for times and resource needs.")
        st.plotly_chart(charts.resources(p, results[a]), width="stretch")
        check = audit(p, results[a].starts)
        (st.success if check.feasible else st.error)(
            f"Independent check: {len(check.precedence_violations)} precedence violations and "
            f"{len(check.resource_violations)} capacity violations — "
            + ("the schedule is valid." if check.feasible else "the schedule is NOT valid."))

# ── 4 Results (kept separate from everything else) ───────────────────────────────
with tabs[3]:
    guide("results")
    if results is None:
        st.info(NEED_RUN)
    elif len(results) < 2:
        st.info("A winner needs a comparison — choose **Run all algorithms** in the sidebar.")
    else:
        e = explain(p, results)
        w = results[e["winner"]]
        gap = 100 * (w.makespan - p.lower_bound) / p.lower_bound
        st.markdown(f"""
<div class="winner"><div class="t">Winner of this run</div>
  <div class="w">🏆 {e['winner']} — {FULL_NAMES[e['winner']]}</div>
  <div class="s">Project length <b>{w.makespan} periods</b> · lower bound {p.lower_bound} · gap {gap:.1f}%
  {'· <b>provably optimal</b>' if e['optimal_proven'] else ''} · decided by <b>{e['decided_by']}</b></div>
</div>""", unsafe_allow_html=True)
        st.markdown(f"**In one sentence:** {e['how']}")
        st.markdown("**How the winner is chosen** (the same rule for every algorithm; names play no part): "
                    + " → ".join(f"**{label}**" for _, label in CRITERIA)
                    + ". A criterion is only used when all earlier ones are tied.")

        st.markdown("##### Final ranking")
        names = {"rank": "Rank", "algorithm": "Algorithm", "name": "Name", "makespan": "Makespan",
                 "gap_to_lower_bound_%": "Gap to lower bound (%)", "behind_winner": "Behind winner",
                 "total_finish": "Total finish time", "levelling": "Levelling", "best_found_at_eval": "Best found at",
                 "mean_evals_5_seeds": "Reliability (avg. evals, 5 seeds)", "evaluations": "Evaluations used",
                 "stopped_early_optimal": "Stopped early (optimal)", "runtime_s": "Runtime (s)", "feasible": "Valid"}
        rank = pd.DataFrame(e["table"]).rename(columns=names)
        rank.insert(3, "Type", ["Proposed" if a in PROPOSED else "Baseline" for a in rank["Algorithm"]])
        rank["Stopped early (optimal)"] = rank["Stopped early (optimal)"].map({True: "Yes", False: "No"})
        rank["Valid"] = rank["Valid"].map({True: "✔", False: "✘"})
        table(rank)

        st.markdown("##### The evidence, step by step")
        first = {a: r.trace[0] for a, r in results.items()}
        st.markdown(
            f"1. **Starting point.** At evaluation 1 the proposed solvers decode the trained GNN's order (makespan "
            f"{', '.join(f'{a} {first[a]}' for a in PROPOSED if a in first)}); the baselines start from random orders "
            f"or a single rule ({', '.join(f'{a} {first[a]}' for a in BASELINES if a in first)}).")
        if e["lead_from_eval"]:
            st.markdown(f"2. **Taking the lead.** From evaluation **{e['lead_from_eval']}** onwards {e['winner']} had a "
                        "strictly better makespan than every other algorithm and kept it to the end.")
        else:
            st.markdown(f"2. **Close race.** At least one other algorithm matched {e['winner']}'s makespan, so the "
                        "tie-break criteria decided.")
        lines = []
        for a, r in results.items():
            if a != e["winner"]:
                at = w.best_trace[min(r.best_eval, len(w.best_trace)) - 1]
                lines.append(f"   - **{a}** reached its best ({r.makespan}) at evaluation {r.best_eval}; "
                             f"by then {e['winner']} already had {at}.")
        st.markdown("3. **Head to head.**\n" + "\n".join(lines))
        st.markdown(f"4. **Schedule quality.** {e['winner']}: total finish time {w.total_finish}, levelling "
                    f"{w.levelling}. TS and PSS keep improving these after reaching their best makespan and stop "
                    "early once the makespan equals the lower bound; the textbook baselines only minimise makespan.")
        st.markdown("5. **Validity.** " + ("All six schedules passed the independent precedence and capacity check."
                    if all(row["feasible"] for row in e["table"]) else "⚠️ Some schedules failed the check."))
        st.download_button("📄  Download PDF report", pdf_report(p, results), f"{p.name}_report.pdf",
                           "application/pdf", type="primary")

# ── 5 GNN inside (plain language) ────────────────────────────────────────────────
with tabs[4]:
    guide("gnn")
    g = gnn.inspect(p)
    if g is None:
        st.warning("No trained model yet — run `python scripts/train.py`.")
    else:
        with st.container(border=True):
            st.markdown(ex.GNN_INTRO)
        st.markdown("**The four steps at a glance**")
        for i, (col, (icon, title, short, _, _)) in enumerate(zip(st.columns(4), ex.GNN_STEPS), 1):
            col.markdown(f'<div class="kpi"><div class="l">{icon} Step {i}</div>'
                         f'<div style="font-weight:700;margin:4px 0">{title}</div><div class="h">{short}</div></div>',
                         unsafe_allow_html=True)
        st.write("")

        def step(i):
            icon, title, short, why, how = ex.GNN_STEPS[i]
            st.markdown(f"#### {icon} Step {i + 1} — {title}")
            c1, c2 = st.columns(2)
            c1.markdown(f"**Why this step?**  \n{why}")
            c2.markdown(f"**How does it do it?**  \n{how}")

        # Step 1 — the facts about each task
        step(0)
        facts = pd.DataFrame(g["features"], columns=[ex.FEATURE_PLAIN[f] for f in g["feature_names"]]).round(2)
        facts.insert(0, "Task", p.task_names)
        busiest = facts.iloc[1:-1].sort_values("All tasks depending on it", ascending=False).iloc[0]
        st.markdown(f"**Example from your project:** *{busiest['Task']}* has many tasks depending on it "
                    f"({busiest['All tasks depending on it']:.0%} of the project) — a fact the network will use "
                    "to push it forward.")
        with st.expander("Show the 10 facts for every task"):
            table(facts, height=300)

        # Step 2 — asking the connected tasks
        step(1)
        pick = st.selectbox("Pick a task to see whom it listens to", range(1, p.n - 1),
                            format_func=lambda i: p.task_names[i])
        for label, view, empty in (("Looking back (tasks before it)", "layer 1 upstream (predecessors)",
                                    "It has no tasks before it — it can start right away."),
                                   ("Looking ahead (tasks after it)", "layer 1 downstream (successors)",
                                    "No tasks are waiting for it.")):
            weights = {p.task_names[j]: float(g["attention"][view][pick][j]) for j in range(p.n)
                       if j != pick and g["attention"][view][pick][j] > 0}
            st.markdown(f"**{label}:**")
            if not weights:
                st.caption(empty)
            for name, wgt in sorted(weights.items(), key=lambda kv: -kv[1]):
                st.progress(min(1.0, wgt), text=f"{name} — {wgt:.0%} of its attention")
        st.caption("The rest of its attention goes to its own facts. A task never looks at tasks it is not connected "
                   "to — the network follows the project's arrows.")
        with st.expander("Show the full attention maps (all tasks, both rounds)"):
            st.caption("Each row is a task, each column a task it may listen to; darker = more attention. "
                       "Empty squares = not connected, so attention is exactly zero.")
            views = list(g["attention"])
            for left, right in zip(views[0::2], views[1::2]):
                c1, c2 = st.columns(2)
                c1.plotly_chart(charts.attention_heatmap(p, g["attention"][left], left), width="stretch")
                c2.plotly_chart(charts.attention_heatmap(p, g["attention"][right], right), width="stretch")

        # Step 3 — the priority list
        step(2)
        queue = pd.DataFrame({"Task": p.task_names, "Classic rule score": g["lst_term"].round(1),
                              "Network's adjustment": g["correction"].round(1), "Final priority": g["score"].round(1),
                              "Queue position (network)": g["rank"], "Queue position (classic rule)": g["lst_rank"]})
        queue = queue[[d > 0 for d in p.durations]].copy()             # real tasks only (no Start/Finish markers)
        for c in ("Queue position (network)", "Queue position (classic rule)"):
            queue[c] = queue[c].rank().astype(int)                     # renumber 1..n over real tasks
        shift = queue["Queue position (classic rule)"] - queue["Queue position (network)"]
        queue["Moved"] = [f"↑ {s}" if s > 0 else f"↓ {-s}" if s < 0 else "" for s in shift]
        queue = queue.sort_values("Queue position (network)")
        moved = int((shift != 0).sum())
        up = queue.loc[shift.idxmax()] if moved else None
        st.markdown(f"**Result:** the network kept the textbook order for {len(queue) - moved} tasks and moved "
                    f"**{moved}** of them" + (f" — e.g. *{up['Task']}* moved up "
                                              f"{int(shift.max())} place{'s' if int(shift.max()) != 1 else ''}." if moved else "."))
        table(queue[["Queue position (network)", "Task", "Moved", "Queue position (classic rule)"]], height=300,
              explain=False)
        with st.expander("Show the scores behind the queue"):
            table(queue, height=300)

        # Step 4 — the schedule and the comparison
        step(3)
        starts, ms = ssgs.decode(p, g["score"].tolist())
        shots = {name.split(" (")[0]: ssgs.decode(p, v)[1] for name, v in rules(p).items()}
        best_rule = min(shots, key=shots.get)
        diff = shots[best_rule] - ms
        pl = lambda k: f"{k} period{'s' if k != 1 else ''}"
        verdict = (f"**{pl(diff)} shorter** than the best textbook rule ({best_rule}, {shots[best_rule]})" if diff > 0
                   else f"the same length as the best textbook rule ({best_rule})" if diff == 0
                   else f"{pl(-diff)} longer than the best textbook rule ({best_rule}, {shots[best_rule]}) on "
                        "this project — the search that follows has to make up the difference")
        st.markdown(f"**Result:** the network's order gives a project of **{ms} periods** — {verdict}, "
                    "before any searching has happened.")
        st.plotly_chart(charts.one_shot_comparison({"GNN (trained)": ms, **shots}, p.lower_bound), width="stretch")
        with st.expander("Show the timetable built from the network's order"):
            st.plotly_chart(charts.gantt(p, Result("GNN", ms, starts, 1, 1, 0.0, [ms], [ms])), width="stretch")
        st.info("**What happens next:** this schedule is attempt no. 1 of Tabu Search and Particle Swarm Search. "
                "They take it from here and keep improving it — see the Iterations and Results tabs.")

# ── 6 Benchmark evidence ─────────────────────────────────────────────────────────
with tabs[5]:
    guide("evidence")
    for title, f in (("Held-out benchmark (260 projects)", ROOT / "docs" / "RESULTS.md"),
                     ("Realistic use cases (66 projects)", ROOT / "docs" / "USE_CASES.md")):
        if f.exists():
            with st.expander(title, expanded=title.startswith("Held")):
                st.markdown(f.read_text())

st.markdown('<div class="foot">AI-RCPSP — built by the project team · B.Tech final-year project, RIET · '
            'data: PSPLIB, RG30 · every explanation on these pages lives in explanation.py</div>', unsafe_allow_html=True)
