"""
03_descriptives.py
Descriptive statistics, tables and figures for the Data section of Paper 1.

Input : Dataset/merged/asrs_merged_dedup.csv
Output: results/descriptives/  (CSV tables + PNG figures)

Tables
  T1 reports per year                     T5 detector distribution
  T2 top anomaly categories               T6 result distribution
  T3 primary problem distribution         T7 reporter function (pilot vs ATC)
  T4 flight phase distribution            T8 field-population summary
Figures
  F1 reports per year        F3 primary problem (top 15)
  F2 narrative length hist   F4 detector distribution
  F5 flight phase (top 12)   F6 When Detected population by year (RQ4 scope)
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repository root. Override with the ASRS_BASE environment variable, e.g.
#   set ASRS_BASE=D:\my\data\dir      (Windows)
#   export ASRS_BASE=/home/me/data     (Linux/macOS)
# Default: the parent of this code/ directory, i.e. the repository root.
BASE = os.environ.get(
    "ASRS_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
IN   = os.path.join(BASE, "Dataset", "merged", "asrs_merged_dedup.csv")
OUT  = os.path.join(BASE, "results", "descriptives")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(IN, dtype=str, low_memory=False)
n = len(df)
print(f"Reports: {n}")

def find(*kw):
    for c in df.columns:
        if all(k.lower() in c.lower() for k in kw):
            return c
    return None

def split_counts(col, sep=";"):
    s = df[col].fillna("").astype(str).str.split(rf"\s*{sep}\s*").explode().str.strip()
    return s[s != ""].value_counts()

def save_tbl(obj, name):
    p = os.path.join(OUT, name)
    obj.to_csv(p)
    print(f"  saved {name}")

plt.rcParams.update({"figure.dpi": 150, "font.size": 9})

# ---- year variable ----
date_col = find("date") or df.columns[1]
yr = df[date_col].fillna("").astype(str).str.extract(r"(\d{4})")[0]
df["_year"] = pd.to_numeric(yr, errors="coerce")

# T1 / F1: reports per year
per_year = df["_year"].value_counts().sort_index()
save_tbl(per_year.rename("reports"), "T1_reports_per_year.csv")
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(per_year.index, per_year.values, color="#33668c", width=0.8)
ax.set_xlabel("Year"); ax.set_ylabel("Reports")
ax.set_title("ATC-related ASRS reports per year (n=30,410)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "F1_reports_per_year.png")); plt.close(fig)

# T2: anomaly categories
anom = find("anomaly")
save_tbl(split_counts(anom).rename("reports"), "T2_anomaly_categories.csv")

# T3 / F3: primary problem
pp = find("primary", "problem")
pp_counts = df[pp].fillna("").str.strip().replace("", pd.NA).dropna().value_counts()
save_tbl(pp_counts.rename("reports"), "T3_primary_problem.csv")
fig, ax = plt.subplots(figsize=(6, 4))
pp_counts.head(15)[::-1].plot.barh(ax=ax, color="#33668c")
ax.set_xlabel("Reports"); ax.set_title("Primary Problem (top 15)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "F3_primary_problem.png")); plt.close(fig)

# T4 / F5: flight phase (semicolon-multi)
fp = find("flight", "phase")
fp_counts = split_counts(fp)
save_tbl(fp_counts.rename("reports"), "T4_flight_phase.csv")
fig, ax = plt.subplots(figsize=(6, 3.5))
fp_counts.head(12)[::-1].plot.barh(ax=ax, color="#33668c")
ax.set_xlabel("Reports"); ax.set_title("Flight Phase (Aircraft 1, top 12)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "F5_flight_phase.png")); plt.close(fig)

# T5 / F4: detector
det = find("detector")
det_counts = split_counts(det)
save_tbl(det_counts.rename("reports"), "T5_detector.csv")
fig, ax = plt.subplots(figsize=(6, 3.5))
det_counts.head(12)[::-1].plot.barh(ax=ax, color="#8c3346")
ax.set_xlabel("Reports"); ax.set_title("Detector — who/what detected the event (top 12)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "F4_detector.png")); plt.close(fig)

# T6: result
res = find("result")
save_tbl(split_counts(res).rename("reports"), "T6_result.csv")

# T7: reporter function
fn = find("Person 1", "Function")
save_tbl(split_counts(fn).rename("reports"), "T7_reporter_function.csv")

# T8: field population summary (key analysis fields)
keys = {
    "Narrative": find("narrative"), "Synopsis": find("synopsis"),
    "Anomaly": anom, "Primary Problem": pp,
    "Human Factors": find("human", "factors"),
    "Contributing Factors": find("contributing"),
    "Detector": det, "When Detected": find("when", "detected"),
    "Result": res, "Miss Distance": find("miss", "distance"),
    "Flight Phase": fp, "Reporter Function": fn,
}
rows = []
for label, col in keys.items():
    s = df[col].fillna("").astype(str).str.strip()
    rows.append({"field": label, "column": col,
                 "non_blank": int((s != "").sum()),
                 "pct": round(100 * (s != "").sum() / n, 1)})
save_tbl(pd.DataFrame(rows).set_index("field"), "T8_field_population.csv")

# F2: narrative length
narr = keys["Narrative"]
lens = df[narr].fillna("").astype(str).str.split().apply(len)
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(lens[lens > 0], bins=60, color="#33668c")
ax.set_xlabel("Narrative length (words)"); ax.set_ylabel("Reports")
ax.set_title(f"Narrative length (median {lens[lens>0].median():.0f} words)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "F2_narrative_length.png")); plt.close(fig)

# F6: When Detected population by year -> justifies RQ4 timing scope (2010+)
wd = keys["When Detected"]
wd_pop = (df[wd].fillna("").astype(str).str.strip() != "")
by_year = df.assign(pop=wd_pop).groupby("_year")["pop"].mean().mul(100)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(by_year.index, by_year.values, color="#8c3346", lw=1.5)
ax.axvline(2010, color="gray", ls="--", lw=0.8)
ax.set_xlabel("Year"); ax.set_ylabel("% populated")
ax.set_title("'When Detected' field population by year (timing analysis scoped to 2010+)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "F6_when_detected_population.png")); plt.close(fig)

print("\nAll tables and figures written to:", OUT)
