"""
02_profile_asrs.py
Profile the merged ASRS dataset for the fields Paper 1 depends on:
  - temporal coverage
  - text fields (Narrative, Synopsis)
  - validation-label fields (Anomaly, Primary Problem, Human Factors)
  - RQ3 detection fields (Detector, When Detected, Result) incl. population by era
  - severity proxy (Miss Distance) and cross-tab fields (Flight Phase, Make/Model)

Input : Dataset/merged/asrs_merged_dedup.csv
Output: console report
"""
import os
import pandas as pd

# Repository root. Override with the ASRS_BASE environment variable, e.g.
#   set ASRS_BASE=D:\my\data\dir      (Windows)
#   export ASRS_BASE=/home/me/data     (Linux/macOS)
# Default: the parent of this code/ directory, i.e. the repository root.
BASE = os.environ.get(
    "ASRS_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
OUT_DIR = os.path.join(BASE, "Dataset", "merged")
df = pd.read_csv(os.path.join(OUT_DIR, "asrs_merged_dedup.csv"), dtype=str, low_memory=False)
n = len(df)
print(f"Unique reports: {n}\n")

def find(*keywords):
    """Return first column whose name contains all keywords (case-insensitive)."""
    for c in df.columns:
        cl = c.lower()
        if all(k.lower() in cl for k in keywords):
            return c
    return None

def fill(col):
    if col is None: return None
    s = df[col].fillna("").astype(str).str.strip()
    nonblank = (s != "").sum()
    return nonblank, 100*nonblank/n

# ---- Temporal coverage ----
date_col = find("date") or df.columns[1]
dates = df[date_col].fillna("").astype(str).str.extract(r"(\d{6})")[0]
yr = dates.str[:4]
print("=== TEMPORAL COVERAGE (by year) ===")
print(f"Range: {yr.min()} to {yr.max()}   |  valid dates: {yr.notna().sum()}/{n}")
dec = (yr.dropna().astype(int)//10*10)
print("\nReports per decade:")
for d, c in dec.value_counts().sort_index().items():
    print(f"  {d}s: {c:6d}")

# ---- Key fields ----
targets = {
    "Narrative (text)":        find("narrative"),
    "Synopsis (text)":         find("synopsis"),
    "Callback (text)":         find("callback"),
    "Anomaly (label)":         find("anomaly"),
    "Human Factors (label)":   find("human", "factors"),
    "Primary Problem":         find("primary", "problem"),
    "Contributing Factors":    find("contributing"),
    "Detector (RQ3)":          find("detector"),
    "When Detected (RQ3)":     find("when", "detected"),
    "Result (RQ3)":            find("result"),
    "Miss Distance (severity proxy)": find("miss", "distance"),
    "Flight Phase":            find("flight", "phase"),
    "Make Model Name":         find("make", "model"),
}
print("\n=== FIELD POPULATION (non-blank %) ===")
for label, col in targets.items():
    if col is None:
        print(f"  {label:34s} -> COLUMN NOT FOUND")
        continue
    nb, pct = fill(col)
    print(f"  {label:34s} {pct:5.1f}%  ({nb}/{n})   [{col}]")

# ---- Narrative length ----
narr = find("narrative")
if narr:
    lens = df[narr].fillna("").astype(str).str.split().apply(len)
    print(f"\n=== NARRATIVE LENGTH (words) ===")
    print(f"  non-empty: {(lens>0).sum()}   median: {lens[lens>0].median():.0f}   "
          f"mean: {lens[lens>0].mean():.0f}   max: {lens.max()}")

# ---- RQ3 detection fields by decade (critical dependency check) ----
det = find("detector"); whend = find("when","detected"); res = find("result")
print("\n=== RQ3 DETECTION FIELDS POPULATION BY DECADE ===")
for label, col in [("Detector", det), ("When Detected", whend), ("Result", res)]:
    if col is None: continue
    s = df[col].fillna("").astype(str).str.strip()
    print(f"\n  {label}:")
    for d in sorted(dec.unique()):
        mask = (yr.dropna().astype(int)//10*10 == d)
        idx = mask[mask].index
        sub = s.loc[idx]
        if len(sub):
            print(f"    {d}s: {100*(sub!='').sum()/len(sub):5.1f}% populated  (n={len(sub)})")

# ---- Top anomaly categories ----
anom = find("anomaly")
if anom:
    print("\n=== TOP 15 ANOMALY VALUES (semicolon-split) ===")
    exploded = df[anom].fillna("").astype(str).str.split(r"\s*;\s*").explode().str.strip()
    exploded = exploded[exploded != ""]
    for v, c in exploded.value_counts().head(15).items():
        print(f"  {c:6d}  {v[:70]}")

print("\n=== done profile ===")
