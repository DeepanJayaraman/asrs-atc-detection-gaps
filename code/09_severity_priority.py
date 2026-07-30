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

# scatter: prevalence vs trend, sized by evasive share, colored by auto-detect
fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
sc = ax.scatter(tbl["mean_share_pct"], tbl["slope_pp_per_decade"],
                s=40 + 900 * tbl["evasive_action_share"].fillna(0),
                c=tbl["auto_detect_share"].fillna(0), cmap="RdYlGn", alpha=0.8)
for t, row in tbl.head(10).iterrows():
    ax.annotate(t[:24], (row["mean_share_pct"], row["slope_pp_per_decade"]),
                fontsize=6, xytext=(4, 3), textcoords="offset points")
ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel("Mean share of reports (%)"); ax.set_ylabel("Trend (pp/decade)")
ax.set_title("Priority signals: prevalence vs trend\n"
             "(size = evasive-action share; color = automation-detection share)",
             fontsize=9)
fig.colorbar(sc, label="auto-detect share")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "F_priority_scatter.png"))
print("saved outputs to", OUT)
