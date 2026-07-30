"""
06_operational_profiles.py  (RQ2 — operational risk factors)

Links hazard topics to operational context:
  - flight phase (multi-label)
  - reporter group (pilot vs controller position) derived from Person 1 Function
  - human factors (multi-label; 54% populated — reported with that caveat)
Chi-square + Cramer's V per dimension; row-normalized profile heatmaps.
Outputs: results/rq2_operational/
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
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
OUT  = os.path.join(BASE, "results", "rq2_operational")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(IN, dtype={"acn": str}, low_memory=False)
df = df[df["topic"] != 6]                      # drop metadata topic
print(f"{len(df)} docs")

PILOT = {"pilot flying", "pilot not flying", "captain", "first officer",
         "single pilot", "instructor", "trainee", "check pilot", "relief pilot",
         "flight engineer"}
CTRL  = {"enroute", "approach", "local", "departure", "ground",
         "supervisor / cic", "handoff / assist", "oceanic", "flight data",
         "coordinator", "traffic management"}

def reporter_group(fn):
    if not isinstance(fn, str) or not fn.strip():
        return None
    toks = {t.strip().lower() for t in fn.split(";")}
    p = bool(toks & PILOT); c = bool(toks & CTRL)
    if p and not c: return "Pilot"
    if c and not p: return "Controller"
    if p and c:     return "Mixed"
    return "Other"

df["reporter"] = df["function"].map(reporter_group)
print(df["reporter"].value_counts(dropna=False).to_string())

def exploded(col):
    e = df[["topic_label", col]].dropna()
    e[col] = e[col].astype(str).str.split(r"\s*;\s*")
    e = e.explode(col).reset_index(drop=True)
    e[col] = e[col].str.strip()
    return e[e[col] != ""]

def analyze(ct, name, top=15, figsize=(12, 9)):
    ct = ct.loc[:, ct.sum().sort_values(ascending=False).head(top).index]
    chi2, p, dof, _ = chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    print(f"{name}: chi2={chi2:.0f} p={p:.1e} CramersV={v:.3f}")
    ct.to_csv(os.path.join(OUT, f"crosstab_{name}.csv"))
    norm = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    im = ax.imshow(norm.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(norm.shape[1]))
    ax.set_xticklabels([c[:30] for c in norm.columns], rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(norm.shape[0]))
    ax.set_yticklabels([r[:45] for r in norm.index], fontsize=6.5)
    ax.set_title(f"Topic x {name} (row-normalized); Cramer's V={v:.2f}", fontsize=9)
    fig.colorbar(im, shrink=0.6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"F_{name}.png")); plt.close(fig)
    return v

metrics = {}
e = exploded("phase")
metrics["flight_phase"] = analyze(pd.crosstab(e["topic_label"], e["phase"]), "flight_phase")

r = df.dropna(subset=["reporter"])
r = r[r["reporter"].isin(["Pilot", "Controller"])]
metrics["reporter"] = analyze(pd.crosstab(r["topic_label"], r["reporter"]),
                              "reporter_group", top=2, figsize=(5, 9))

e = exploded("hf")
metrics["human_factors"] = analyze(pd.crosstab(e["topic_label"], e["hf"]), "human_factors")

# pilot-vs-controller share per topic (a headline RQ2 table)
share = (pd.crosstab(r["topic_label"], r["reporter"], normalize="index")
         .sort_values("Controller", ascending=False).round(3))
share.to_csv(os.path.join(OUT, "topic_reporter_share.csv"))

with open(os.path.join(OUT, "metrics.txt"), "w") as fh:
    for k, v in metrics.items():
        fh.write(f"Cramer's V topic x {k}: {v:.3f}\n")
print("saved outputs to", OUT)
