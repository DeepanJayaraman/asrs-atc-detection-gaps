"""
05_validate_topics.py  (RQ1 — structure & convergent validation)

1. Assigns provisional human-readable labels to the 36 BERTopic topics and
   merges near-duplicates (T29 -> T18, both TCAS RA).
2. Convergent validation against NASA-analyst labels:
     - topic x Primary Problem contingency: chi-square, Cramer's V, NMI
     - topic x Anomaly (multi-label, exploded): row-normalized heatmap
     - per-topic modal analyst label + purity (share of docs matching modal label)
3. Outputs (results/rq1_validation/):
     doc_topics_labeled.csv, topic_labels.csv, alignment_primary_problem.csv,
     alignment_anomaly.csv, validation_metrics.txt, F_heatmap_anomaly.png,
     F_heatmap_primary.png
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import normalized_mutual_info_score
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
IN   = os.path.join(BASE, "results", "topic_model", "doc_topics.csv")
OUT  = os.path.join(BASE, "results", "rq1_validation")
os.makedirs(OUT, exist_ok=True)

# Provisional labels (from topics_top_words.txt, run 2). Refine at writing time.
MERGE = {29: 18}                      # TCAS RA duplicates
EXCLUDE = {6}                         # aircraft-class metadata, not a hazard
LABELS = {
    0: "Ground/taxi conflict (hold short)",
    1: "Sector/airspace management workload",
    2: "Enroute separation (controller reports, legacy)",
    3: "Arrival procedure/restriction (STAR) deviations",
    4: "Traffic-pattern conflicts (tower/GA)",
    5: "MVA low-altitude alerts (vectoring)",
    6: "(aircraft-class metadata - excluded)",
    7: "Local control / vehicle-runway operations",
    8: "Radio communication breakdown",
    9: "Route clearance / FMS routing errors",
    10: "Weather deviation / turbulence encounter",
    11: "VFR-IFR mixed-traffic conflicts",
    12: "Wake turbulence encounters",
    13: "Takeoff position/clearance (controller reports, legacy)",
    14: "Altitude-clearance deviation (level bust)",
    15: "Fuel/dispatch emergencies",
    16: "Helicopter operations",
    17: "Terrain warnings (GPWS)",
    18: "TCAS RA conflicts",
    19: "Class B/C airspace incursions",
    20: "Enroute separation events (type-coded)",
    21: "Visual separation on final",
    22: "Landing clearance (tower)",
    23: "Conflict-alert activations (ATC automation)",
    24: "NOTAM / airport-status information failures",
    25: "Glideslope capture issues",
    26: "NMAC / evasive action (visual)",
    27: "Takeoff clearance events",
    28: "SID departure deviations",
    30: "Heading/vector compliance",
    31: "Level bust with separation loss",
    32: "Visual approach clearances",
    33: "Wind/windshear on landing",
    34: "MSAW / minimum-IFR-altitude alerts",
    35: "Localizer intercept issues",
}

df = pd.read_csv(IN, dtype={"acn": str}, low_memory=False)
df["topic"] = df["topic"].replace(MERGE)
df["topic_label"] = df["topic"].map(LABELS)
n_all = len(df)
dfh = df[~df["topic"].isin(EXCLUDE)].copy()      # hazard topics only
print(f"{n_all} docs; {len(dfh)} in hazard topics ({len(set(dfh['topic']))} topics)")

# ---- topic sizes ----
sizes = dfh.groupby(["topic", "topic_label"]).size().rename("n").reset_index()
sizes.to_csv(os.path.join(OUT, "topic_labels.csv"), index=False)
df.to_csv(os.path.join(OUT, "doc_topics_labeled.csv"), index=False)

# ---- validation vs Primary Problem (single label) ----
v = dfh.dropna(subset=["primary"]).copy()
v = v[v["primary"].str.strip() != ""]
ct = pd.crosstab(v["topic_label"], v["primary"])
chi2, p, dof, _ = chi2_contingency(ct)
cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
nmi = normalized_mutual_info_score(v["topic"], v["primary"])
print(f"Primary Problem: chi2={chi2:.0f} (p={p:.2e}), Cramer's V={cramers_v:.3f}, NMI={nmi:.3f}")
ct.to_csv(os.path.join(OUT, "alignment_primary_problem.csv"))

# ---- validation vs Anomaly (multi-label, exploded; drop the universal filter tag) ----
a = dfh.dropna(subset=["anomaly"]).copy()
a["anom"] = a["anomaly"].str.split(r"\s*;\s*")
a = a.explode("anom").reset_index(drop=True)
a["anom"] = a["anom"].str.strip()
a = a[(a["anom"] != "") & (a["anom"] != "ATC Issue All Types")]
top_anoms = a["anom"].value_counts().head(20).index
cta = pd.crosstab(a.loc[a["anom"].isin(top_anoms), "topic_label"],
                  a.loc[a["anom"].isin(top_anoms), "anom"])
cta.to_csv(os.path.join(OUT, "alignment_anomaly.csv"))

# ---- per-topic modal label & purity ----
rows = []
for t, g in v.groupby("topic_label"):
    modal = g["primary"].mode().iloc[0]
    purity = (g["primary"] == modal).mean()
    rows.append({"topic": t, "n": len(g), "modal_primary_problem": modal,
                 "purity": round(purity, 3)})
pur = pd.DataFrame(rows).sort_values("n", ascending=False)
pur.to_csv(os.path.join(OUT, "topic_modal_primary.csv"), index=False)

with open(os.path.join(OUT, "validation_metrics.txt"), "w") as fh:
    fh.write(f"Docs in hazard topics: {len(dfh)} of {n_all}\n"
             f"Topics (after merge/exclude): {len(set(dfh['topic']))}\n\n"
             f"Topic x Primary Problem: chi2={chi2:.1f}, dof={dof}, p={p:.3e}\n"
             f"Cramer's V = {cramers_v:.3f}\nNMI = {nmi:.3f}\n\n"
             f"Mean topic purity (modal primary problem): {pur['purity'].mean():.3f}\n")

# ---- heatmaps (row-normalized) ----
def heatmap(ct_, fname, title, figsize=(13, 10)):
    norm = ct_.div(ct_.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    im = ax.imshow(norm.values, aspect="auto", cmap="Blues", vmin=0, vmax=0.6)
    ax.set_xticks(range(norm.shape[1]))
    ax.set_xticklabels([c[:38] for c in norm.columns], rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(norm.shape[0]))
    ax.set_yticklabels([r[:45] for r in norm.index], fontsize=6.5)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, shrink=0.6, label="row share")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname)); plt.close(fig)

heatmap(cta, "F_heatmap_anomaly.png",
        "Topic x Anomaly (NASA analyst labels; row-normalized)")
top_pp = ct.loc[:, ct.sum().sort_values(ascending=False).head(15).index]
heatmap(top_pp, "F_heatmap_primary.png",
        "Topic x Primary Problem (top 15; row-normalized)")
print("saved outputs to", OUT)
