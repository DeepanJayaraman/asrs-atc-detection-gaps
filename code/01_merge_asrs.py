"""
01_merge_asrs.py
Merge 8 NASA ASRS DBOnline exports into one deduplicated dataset.

File layout (tab-separated, .xls extension but actually TSV):
  row 1 : group header (Time, Place, Environment, Aircraft 1, ...)
  row 2 : field names (ACN, Date, Local Time Of Day, ...)
  row 3 : blank
  row 4+: data
Several field names repeat (Aircraft 1 / Aircraft 2, Person 1 / Person 2),
so we build unique column names by combining group header + field name and
disambiguating any remaining collisions with a numeric suffix.

Input : Dataset/*.xls   (8 files)
Output: Dataset/merged/asrs_merged_dedup.csv
        Dataset/merged/columns.txt
"""
import glob, os, csv
import pandas as pd

# Repository root. Override with the ASRS_BASE environment variable, e.g.
#   set ASRS_BASE=D:\my\data\dir      (Windows)
#   export ASRS_BASE=/home/me/data     (Linux/macOS)
# Default: the parent of this code/ directory, i.e. the repository root.
BASE = os.environ.get(
    "ASRS_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA_DIR = os.path.join(BASE, "Dataset")
OUT_DIR  = os.path.join(BASE, "Dataset", "merged")
os.makedirs(OUT_DIR, exist_ok=True)

def build_columns(group_row, name_row):
    """Combine group + field name into unique column labels."""
    cols, seen = [], {}
    for g, n in zip(group_row, name_row):
        g = (g or "").strip()
        n = (n or "").strip()
        base = n if n else g
        # prefix Aircraft/Person/Component groups so repeated field names stay distinct
        if g and n and g not in ("Time", "Place", "Environment"):
            label = f"{g} | {n}" if g not in n else n
        else:
            label = base
        if label in seen:
            seen[label] += 1
            label = f"{label} ({seen[label]})"
        else:
            seen[label] = 0
        cols.append(label)
    return cols

frames = []
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.xls")))
print(f"Found {len(files)} files\n")

for path in files:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        rows = list(reader)
    group_row = rows[0]
    name_row  = rows[1]
    # data starts at row index 3 (row 4); row index 2 is blank
    data_rows = [r for r in rows[3:] if any(c.strip() for c in r)]
    cols = build_columns(group_row, name_row)
    # normalize row length to number of columns
    fixed = [ (r + [""] * len(cols))[:len(cols)] for r in data_rows ]
    df = pd.DataFrame(fixed, columns=cols)
    df["__source_file"] = os.path.basename(path)
    frames.append(df)
    print(f"{os.path.basename(path):28s} rows={len(df):6d}")

raw = pd.concat(frames, ignore_index=True)
print(f"\nTotal rows before dedup: {len(raw)}")

# ACN is the unique report id (first column).
acn_col = raw.columns[0]
print(f"ACN column detected as: '{acn_col}'")
raw[acn_col] = raw[acn_col].astype(str).str.strip()

# Keep only rows with a valid numeric ACN
valid = raw[raw[acn_col].str.match(r"^\d+$", na=False)].copy()
print(f"Rows with valid numeric ACN: {len(valid)}")

# Dedup: keep first occurrence of each ACN
dedup = valid.drop_duplicates(subset=[acn_col], keep="first").reset_index(drop=True)
print(f"Unique reports after dedup on ACN: {len(dedup)}")
print(f"Duplicates removed: {len(valid) - len(dedup)}")

out_csv = os.path.join(OUT_DIR, "asrs_merged_dedup.csv")
dedup.to_csv(out_csv, index=False, encoding="utf-8")
print(f"\nSaved: {out_csv}")

with open(os.path.join(OUT_DIR, "columns.txt"), "w", encoding="utf-8") as fh:
    for i, c in enumerate(dedup.columns):
        fh.write(f"{i:3d}\t{c}\n")

print("\n=== done merge ===")
