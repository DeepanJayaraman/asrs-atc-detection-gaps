"""
08_detection.py  (RQ4 — detection pathways; the paper's headline analysis)

Who catches each hazard theme, when, and with what outcome:
  - Detector grouped: Controller / Flight crew / Aircraft automation /
    ATC automation / Other  (full history 1988-2026)
  - When Detected (2010+ only; field unpopulated before)
  - Result categories per topic
  - Detection-gap indicators per topic:
      auto_share  = share detected by automation (either side)
      late_share  = share detected in flight vs pre-flight/taxi (2010+)
      evasive     = share of results involving evasive action (severity signal)
Outputs: results/rq4_detection/
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
IN   = os.path.join(BASE, "results", "rq1_validation", "doc_topics_labeled.csv")
OUT  = os.path.join(BASE, "results", "rq4_detection")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(IN, dtype={"acn": str}, low_memory=False)
df = df[df["topic"] != 6]
print(f"{len(df)} docs")

def det_group(d):
    if not isinstance(d, str) or not d.strip():
        return None
    d = d.lower()
    if "automation" in d:
        return "ATC automation" if "air traffic control" in d else "Aircraft automation"
    if "flight crew" in d:            return "Flight crew"
    if "air traffic control" in d:    return "Controller"
    return "Other person"

# multi-label detector: explode, group, dedupe per report
e = df[["acn", "topic_label", "detector"]].dropna()
e["det"] = e["detector"].astype(str).str.split(r"\s*;\s*")
e = e.explode("det").reset_index(drop=True)
e["grp"] = e["det"].map(det_group)
e = e.dropna(subset=["grp"]).drop_duplicates(["acn", "grp"])

ct = pd.crosstab(e["topic_label"], e["grp"])
ct.to_csv(os.path.join(OUT, "crosstab_detector.csv"))
norm = ct.div(ct.sum(axis=1), axis=0)

order = norm.sort_values("Controller", ascending=True).index
fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
bottom = np.zeros(len(order))
colors = {"Controller": "#33668c", "Flight crew": "#8c3346",
          "Aircraft automation": "#e0a030", "ATC automation": "#4a9a70",
          "Other person": "#999999"}
for g in ["Controller", "Flight crew", "Aircraft automation", "ATC automation", "Other person"]:
    if g in norm.columns:
        vals = norm.loc[order, g].values
        ax.barh(range(len(order)), vals, left=bottom, color=colors[g], label=g)
        bottom += vals
ax.set_yticks(range(len(order)))
ax.set_yticklabels([t[:45] for t in order], fontsize=6.5)
ax.set_xlabel("Share of reports"); ax.legend(fontsize=7, loc="lower right")
ax.set_title("Who detects each hazard theme (1988-2026)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "F_detector_stacked.png")); plt.close(fig)

# ---- When Detected (2010+) ----
w = df[(df["year"] >= 2010) & df["when"].notna()].copy()
w["when"] = w["when"].astype(str).str.strip()
w = w[w["when"] != ""]
print("When Detected values:", w["when"].value_counts().head(10).to_dict())
ctw = pd.crosstab(w["topic_label"], w["when"])
ctw.to_csv(os.path.join(OUT, "crosstab_when_detected_2010plus.csv"))

# ---- Result ----
r = df[["acn", "topic_label", "result"]].dropna()
r["res"] = r["result"].astype(str).str.split(r"\s*;\s*")
r = r.explode("res").reset_index(drop=True)
r["res"] = r["res"].str.strip()
r = r[r["res"] != ""]
ctr = pd.crosstab(r["topic_label"], r["res"])
ctr.to_csv(os.path.join(OUT, "crosstab_result.csv"))

# ---- detection-gap indicator table ----
auto_share = (norm.get("Aircraft automation", 0) + norm.get("ATC automation", 0))
inflight_terms = ctw.columns[ctw.columns.str.contains("In-flight", case=False)]
late_share = ctw[inflight_terms].sum(axis=1) / ctw.sum(axis=1)
evasive = (r.assign(ev=r["res"].str.contains("Evasive", case=False))
             .groupby("topic_label")["ev"].mean())
gap = pd.DataFrame({"auto_detect_share": auto_share.round(3),
                    "inflight_detect_share_2010p": late_share.round(3),
                    "evasive_action_share": evasive.round(3),
                    "n": ct.sum(axis=1)}).sort_values("auto_detect_share")
gap.to_csv(os.path.join(OUT, "detection_gap_indicators.csv"))
print(gap.head(12).to_string())
print("saved outputs to", OUT)
