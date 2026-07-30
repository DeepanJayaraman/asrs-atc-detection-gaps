"""
07_trends.py  (RQ3 — evolution 1988-2026)

Yearly PROPORTIONS per hazard topic (reporting-bias mitigation), then:
  - Mann-Kendall-style monotonic trend: Kendall's tau of proportion vs year
  - Theil-Sen slope (pp per decade)
  - Benjamini-Hochberg correction across topics
  - classification: Emerging / Declining / Stable
2026 is a partial year (data to Jan) -> excluded from trend fits.
Outputs: results/rq3_trends/  (trend table + small-multiple figure)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes
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
OUT  = os.path.join(BASE, "results", "rq3_trends")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(IN, dtype={"acn": str}, low_memory=False)
df = df[(df["topic"] != 6) & df["year"].notna()]
df["year"] = df["year"].astype(int)
df = df[df["year"] <= 2025]                     # 2026 partial
print(f"{len(df)} docs, {df['year'].min()}-{df['year'].max()}")

tot = df.groupby("year").size()
prop = (df.groupby(["year", "topic_label"]).size()
          .unstack(fill_value=0)
          .div(tot, axis=0))

rows = []
for t in prop.columns:
    y = prop[t]
    tau, p = kendalltau(y.index, y.values)
    slope, *_ = theilslopes(y.values, y.index)
    rows.append({"topic": t, "mean_share_pct": round(100 * y.mean(), 2),
                 "tau": round(tau, 3), "p": p,
                 "slope_pp_per_decade": round(1000 * slope, 3)})
res = pd.DataFrame(rows)

# Benjamini-Hochberg (step-up, correctly aligned to rows)
m = len(res)
order = np.argsort(res["p"].values)
q = np.empty(m)
prev = 1.0
for rank_pos in range(m - 1, -1, -1):
    i = order[rank_pos]
    prev = min(prev, res["p"].values[i] * m / (rank_pos + 1))
    q[i] = prev
res["q"] = q
res["trend"] = np.where(res["q"] >= 0.05, "Stable",
                np.where(res["tau"] > 0, "Emerging", "Declining"))
# Era-style topics: trend reflects ASRS writing-style change, not hazard change.
LEGACY = ["Enroute separation (controller reports, legacy)",
          "Takeoff position/clearance (controller reports, legacy)",
          "Enroute separation events (type-coded)"]
res.loc[res["topic"].isin(LEGACY), "trend"] = "Era-artifact (excluded)"
res = res.sort_values("slope_pp_per_decade", ascending=False)
res.to_csv(os.path.join(OUT, "trend_table.csv"), index=False)
print(res[["topic", "mean_share_pct", "tau", "q", "slope_pp_per_decade", "trend"]]
      .to_string(index=False))

prop.to_csv(os.path.join(OUT, "topic_share_by_year.csv"))

# small multiples: 6 strongest emerging + 6 strongest declining (significant only)
sig = res[res["trend"] != "Stable"]
sel = pd.concat([sig.head(6), sig.tail(6)])["topic"].tolist()
fig, axes = plt.subplots(4, 3, figsize=(11, 10), dpi=150, sharex=True)
for ax, t in zip(axes.flat, sel):
    y = prop[t].mul(100)
    ax.plot(y.index, y.values, lw=0.8, color="#33668c")
    ax.plot(y.index, y.rolling(5, center=True).mean(), lw=1.6, color="#8c3346")
    tr = res.loc[res["topic"] == t, "trend"].iloc[0]
    ax.set_title(f"{t[:40]}\n[{tr}]", fontsize=7)
    ax.tick_params(labelsize=6)
for ax in axes.flat[len(sel):]:
    ax.axis("off")
fig.suptitle("Topic share of ATC-related reports by year (%; red = 5-yr mean)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "F_trends_small_multiples.png"))
print("saved outputs to", OUT)
