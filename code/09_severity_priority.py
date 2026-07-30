"""
09_severity_priority.py  (RQ5 — descriptive severity & priority signals)

Explicitly DESCRIPTIVE (formal MCDM/MOO ranking is Paper 2). Severity proxies:
  - Miss Distance (numeric, ~39% populated, mostly conflict events) — disclosed
  - evasive-action share (from RQ4)
Combines with prevalence (mean share) and trend slope (RQ3) into a
priority-signal table: topics that are frequent AND rising AND weakly
auto-detected AND severe-proxied.
Outputs: results/rq5_priority/priority_table.csv (+ scatter figure)
"""
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repository root. Override with the ASRS_BASE environment variable, e.g.
#   set ASRS_BASE=D:\my\data\dir      (Windows)
#   export ASRS_BASE=/home/me/data     (Linux/macOS)
# Default: the parent of this code/ directory, i.e. the repository root.
BASE = os.environ.get(
    "ASRS_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
LAB  = os.path.join(BASE, "results", "rq1_validation", "doc_topics_labeled.csv")
TRND = os.path.join(BASE, "results", "rq3_trends", "trend_table.csv")
GAP  = os.path.join(BASE, "results", "rq4_detection", "detection_gap_indicators.csv")
OUT  = os.path.join(BASE, "results", "rq5_priority")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(LAB, dtype={"acn": str}, low_memory=False)
df = df[df["topic"] != 6]

# Miss Distance is structured text: "Horizontal 2500; Vertical 200" (feet).
# Parse both components; use HORIZONTAL as the primary proxy (vertical as backup).
merged = pd.read_csv(os.path.join(BASE, "Dataset", "merged", "asrs_merged_dedup.csv"),
                     dtype=str, low_memory=False)
mcol = [c for c in merged.columns if "Miss Distance" in c][0]
mm = merged[["ACN", mcol]].rename(columns={"ACN": "acn", mcol: "missdist"})
df = df.merge(mm, on="acn", how="left")
df["md_h"] = pd.to_numeric(
    df["missdist"].str.extract(r"Horizontal\s+(\d+)")[0], errors="coerce")
df["md_v"] = pd.to_numeric(
    df["missdist"].str.extract(r"Vertical\s+(\d+)")[0], errors="coerce")
for c in ("md_h", "md_v"):
    df.loc[(df[c] <= 0) | (df[c] > 50000), c] = np.nan
print(f"miss distance: horizontal for {df['md_h'].notna().sum()}, "
      f"vertical for {df['md_v'].notna().sum()} reports")

sev = df.groupby("topic_label").agg(
    n=("acn", "size"),
    md_n=("md_h", "count"),
    median_miss_ft=("md_h", "median"),
    median_vert_ft=("md_v", "median")).round(0)

trend = pd.read_csv(TRND).set_index("topic")
gap = pd.read_csv(GAP, index_col=0)

tbl = (sev.join(trend[["mean_share_pct", "slope_pp_per_decade", "trend"]])
          .join(gap[["auto_detect_share", "evasive_action_share"]]))

# priority signal: normalized composite (descriptive only, equal weights;
# formal weighting is Paper 2). Low miss distance & low auto-detect => riskier.
z = lambda s: (s - s.min()) / (s.max() - s.min())
tbl["priority_signal"] = (
    z(tbl["mean_share_pct"].fillna(0))
    + z(tbl["slope_pp_per_decade"].fillna(0))
    + z(tbl["evasive_action_share"].fillna(0))
    + (1 - z(tbl["auto_detect_share"].fillna(tbl["auto_detect_share"].mean())))
    + (1 - z(tbl["median_miss_ft"].fillna(tbl["median_miss_ft"].median())))
).round(3)
tbl = tbl.sort_values("priority_signal", ascending=False)
tbl.to_csv(os.path.join(OUT, "priority_table.csv"))
print(tbl[["n", "mean_share_pct", "slope_pp_per_decade", "trend",
           "auto_detect_share", "evasive_action_share", "median_miss_ft",
           "priority_signal"]].head(12).to_string())

# ------------------------------------------------------------------ figure --
# scatter: prevalence vs trend, sized by evasive share, coloured by auto-detect.
#
# Labelling policy: every theme named in the paper's prose or in the trends
# table is labelled, in addition to the top-ranked themes by priority signal.
# Without this the reader cannot locate themes the text argues about (e.g.
# TCAS RA conflicts, sector workload), which rank outside the top ten.
KEY_THEMES = [
    # named in the RQ3 trends table
    "Sector/airspace management workload",
    "Ground/taxi conflict (hold short)",
    "Traffic-pattern conflicts (tower/GA)",
    "VFR-IFR mixed-traffic conflicts",
    "MVA low-altitude alerts (vectoring)",
    "Local control / vehicle-runway operations",
    "Radio communication breakdown",
    "Landing clearance (tower)",
    "Conflict-alert activations (ATC automation)",
    "Altitude-clearance deviation (level bust)",
    "Heading/vector compliance",
    "TCAS RA conflicts",
    # named in the RQ5 narrative
    "NMAC / evasive action (visual)",
    "Takeoff clearance events",
    "Helicopter operations",
]

# Era-artefact themes are style artefacts, not hazards: their slopes reflect a
# change in reporting orthography. They are shown for completeness (as the
# Methods section states) but drawn in grey and excluded from interpretation.
is_art = tbl["trend"].astype(str).str.contains("rtifact|rtefact", case=False,
                                               na=False)
main_t, art_t = tbl[~is_art], tbl[is_art]

size = lambda s: 40 + 900 * s.fillna(0)

fig, ax = plt.subplots(figsize=(11, 8), dpi=200)
sc = ax.scatter(main_t["mean_share_pct"], main_t["slope_pp_per_decade"],
                s=size(main_t["evasive_action_share"]),
                c=main_t["auto_detect_share"].fillna(0), cmap="RdYlGn",
                vmin=0, vmax=float(tbl["auto_detect_share"].max()),
                alpha=0.85, edgecolors="black", linewidths=0.4, zorder=3)
ax.scatter(art_t["mean_share_pct"], art_t["slope_pp_per_decade"],
           s=size(art_t["evasive_action_share"]),
           facecolors="none", edgecolors="0.45", linewidths=1.2,
           linestyle="--", alpha=0.9, zorder=2)

ax.axhline(0, color="gray", lw=0.6, zorder=1)
# Widen the data limits before placing labels so long theme names have room
# inside the axes instead of running under the colourbar.
ax.margins(x=0.16, y=0.10)

# ---- label placement with greedy collision avoidance ------------------------
# Candidates are tried in order; a label is accepted only if it neither
# overlaps an already-placed label nor spills outside the axes.
CANDS = [(7, 5), (-7, 5), (7, -11), (-7, -11), (12, 13), (-12, 13),
         (12, -19), (-12, -19), (0, 16), (0, -21), (20, 0), (-20, 0),
         (26, 20), (-26, 20), (26, -26), (-26, -26)]

_labels, _markers = [], []      # obstacle sets, shared across placement passes

def _seed_marker_obstacles():
    """Treat every plotted marker as a soft obstacle. Marker area is in
    points^2; convert to a display-space box around the point."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    scale = fig.dpi / 72.0
    for _, row in tbl.iterrows():
        ev = row["evasive_action_share"]
        r = ((40 + 900 * (ev if pd.notna(ev) else 0.0)) / np.pi) ** 0.5 * scale
        px, py = ax.transData.transform((row["mean_share_pct"],
                                         row["slope_pp_per_decade"]))
        _markers.append(Bbox.from_extents(px - r, py - r, px + r, py + r))

def _overlap_area(a, b):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if (w > 0 and h > 0) else 0.0

def place(items, fontsize=7.5):
    """items: (x, y, text, colour). Score every candidate offset and keep the
    best. Label-on-label overlap is weighted far above label-on-marker, since
    text over text is unreadable while text over a circle is merely untidy."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(renderer=rend)
    for x, y, txt, col in items:
        best = None                             # (penalty, bb, dx, dy)
        for dx, dy in CANDS:
            ann = ax.annotate(txt, (x, y), xytext=(dx, dy),
                              textcoords="offset points", fontsize=fontsize,
                              color=col, zorder=5,
                              ha="left" if dx >= 0 else "right",
                              va="bottom" if dy >= 0 else "top")
            bb = ann.get_window_extent(renderer=rend)
            penalty = (200.0 * sum(_overlap_area(bb, o) for o in _labels)
                       + 1.0 * sum(_overlap_area(bb, o) for o in _markers))
            if not (bb.x0 >= axbb.x0 and bb.x1 <= axbb.x1
                    and bb.y0 >= axbb.y0 and bb.y1 <= axbb.y1):
                penalty += 1e9                  # never spill outside the axes
            penalty += 0.02 * (dx * dx + dy * dy)   # prefer nearby placement
            if best is None or penalty < best[0]:
                best = (penalty, bb, dx, dy)
            ann.remove()
            if penalty == 0.02 * (dx * dx + dy * dy):
                break                           # perfectly clear: stop early
        _, bb, dx, dy = best
        ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=fontsize, color=col, zorder=5,
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top")
        _labels.append(bb)
        if abs(dx) > 12 or abs(dy) > 16:        # leader line for distant labels
            ax.annotate("", (x, y), xytext=(dx, dy),
                        textcoords="offset points", zorder=1,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="0.55"))

_seed_marker_obstacles()
labelled = list(dict.fromkeys(
    list(main_t.head(10).index) + [k for k in KEY_THEMES if k in main_t.index]))
place([(main_t.at[t, "mean_share_pct"], main_t.at[t, "slope_pp_per_decade"],
        t, "black") for t in labelled])
place([(art_t.at[t, "mean_share_pct"], art_t.at[t, "slope_pp_per_decade"],
        t.replace(" (controller reports, legacy)", "").replace(
            " (type-coded)", "") + " [era artefact]", "0.4")
       for t in art_t.index], fontsize=7)
ax.set_xlabel("Mean share of reports (%)")
ax.set_ylabel("Trend (percentage points per decade)")
ax.set_title("Priority signals: prevalence against trend\n"
             "(marker size = evasive-action share; "
             "colour = automation-detection share)", fontsize=10)
fig.colorbar(sc, ax=ax, label="automation-detection share", pad=0.02)

# size legend for the evasive-action encoding
handles = [plt.scatter([], [], s=size(pd.Series([v])), facecolor="0.75",
                       edgecolor="black", linewidth=0.4,
                       label=f"{v:.0%}") for v in (0.10, 0.25, 0.45)]
handles.append(plt.scatter([], [], s=90, facecolors="none", edgecolors="0.45",
                           linewidths=1.2, linestyle="--",
                           label="era artefact\n(excluded)"))
ax.legend(handles=handles, title="evasive-action share", loc="lower left",
          fontsize=7.5, title_fontsize=8, labelspacing=1.4,
          borderpad=0.9, framealpha=0.9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "F_priority_scatter.png"), bbox_inches="tight")
print(f"figure: {len(labelled)} themes labelled, "
      f"{len(art_t)} era artefacts marked, {len(tbl)} plotted")
print("saved outputs to", OUT)
