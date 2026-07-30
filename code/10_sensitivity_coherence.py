"""
10_sensitivity_coherence.py
Closes the two analysis-side TODOs in the manuscript:

A. CORE-CLUSTER SENSITIVITY (Limitations item 3 / Appendix)
   Re-derives the deterministic seed-42 UMAP+HDBSCAN solution from cached
   embeddings to recover which documents were CORE cluster members vs
   c-TF-IDF-reassigned outliers, then recomputes the headline results on the
   core subset only:
     - RQ1: Cramer's V / NMI vs Primary Problem
     - RQ4: automation-detection share per theme (rank correlation vs full)
     - RQ3: Theil-Sen slope sign agreement for non-artifact themes

B. TOPIC COHERENCE + SEED STABILITY (Methods TODO)
   - gensim c_v and c_npmi coherence on top-10 words per topic
   - UMAP seeds {1,2,3} re-clustered; agreement with seed-42 solution via
     adjusted Rand index / NMI on jointly-clustered (core) documents

Outputs: results/rq_sensitivity/  (sensitivity_report.txt + CSVs)
"""
import os, re, time
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes, spearmanr, chi2_contingency
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# Repository root. Override with the ASRS_BASE environment variable, e.g.
#   set ASRS_BASE=D:\my\data\dir      (Windows)
#   export ASRS_BASE=/home/me/data     (Linux/macOS)
# Default: the parent of this code/ directory, i.e. the repository root.
BASE = os.environ.get(
    "ASRS_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
TM   = os.path.join(BASE, "results", "topic_model")
OUT  = os.path.join(BASE, "results", "rq_sensitivity")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# ---------- reproduce the document set exactly as in 04 ----------
ABBREV = {
    "acft": "aircraft", "tfc": "traffic", "ctlr": "controller",
    "ctlrs": "controllers", "clrnc": "clearance", "clred": "cleared",
    "clr": "clear", "tkof": "takeoff", "lndg": "landing", "txwy": "taxiway",
    "rwy": "runway", "twr": "tower", "gnd": "ground", "arpt": "airport",
    "hdg": "heading", "alt": "altitude", "dep": "departure", "arr": "arrival",
    "apch": "approach", "dscnt": "descent", "dsnd": "descend",
    "dsnded": "descended", "dsnding": "descending", "clb": "climb",
    "clbed": "climbed", "clbing": "climbing", "flt": "flight",
    "ctr": "center", "freq": "frequency", "freqs": "frequencies",
    "xing": "crossing", "xed": "crossed", "rptr": "reporter",
    "capt": "captain", "plt": "pilot", "mi": "miles", "deg": "degrees",
    "degs": "degrees", "kts": "knots", "ft": "feet", "pos": "position",
    "maint": "maintenance", "emer": "emergency", "eng": "engine",
    "rcvd": "received", "hdof": "handoff", "xmission": "transmission",
    "acr": "air carrier", "sma": "small aircraft", "smt": "small transport",
    "mlg": "medium large transport", "mlt": "medium large transport",
    "lgt": "light transport", "ltt": "light transport", "wdb": "widebody",
    "tcasii": "tcas", "svfr": "special vfr", "intxn": "intersection",
    "vcty": "vicinity", "wx": "weather", "tstorm": "thunderstorm",
    "turb": "turbulence", "vis": "visibility", "rpted": "reported",
    "spd": "speed", "gs": "glideslope", "loc": "localizer",
}
_abbrev_re = re.compile(r"\b(" + "|".join(sorted(ABBREV, key=len, reverse=True)) + r")\b")

def clean(t):
    t = str(t).lower()
    t = re.sub(r"\bcallback\b.*?$", " ", t)
    t = re.sub(r"\[.*?\]", " ", t)
    t = re.sub(r"\b[a-z]{3}\d*\.(airport|tracon|artcc|vor)\b", " ", t)
    t = re.sub(r"\bzzz\w*\b", " ", t)
    t = _abbrev_re.sub(lambda m: ABBREV[m.group(1)], t)
    t = re.sub(r"\bfl\d{3}\b", " flight level ", t)
    return re.sub(r"\s+", " ", t).strip()

merged = pd.read_csv(os.path.join(BASE, "Dataset", "merged", "asrs_merged_dedup.csv"),
                     dtype=str, low_memory=False)
narr_col = [c for c in merged.columns if "Report 1 | Narrative" in c][0]
texts = merged[narr_col].fillna("").map(clean).tolist()
keep = [i for i, t in enumerate(texts) if len(t.split()) >= 20]
docs = [texts[i] for i in keep]
emb = np.load(os.path.join(TM, "embeddings.npy"))
assert emb.shape[0] == len(docs), f"embed/docs mismatch {emb.shape[0]} vs {len(docs)}"
print(f"[{time.time()-t0:5.0f}s] {len(docs)} docs reproduced, embeddings match")

# saved final assignments (post reassignment, pre 05-merge)
dt = pd.read_csv(os.path.join(TM, "doc_topics.csv"), dtype={"acn": str},
                 low_memory=False)
assert len(dt) == len(docs)

from umap import UMAP
from hdbscan import HDBSCAN

def cluster(seed):
    red = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
               random_state=seed).fit_transform(emb)
    return HDBSCAN(min_cluster_size=100, min_samples=10, metric="euclidean",
                   cluster_selection_method="leaf").fit(red).labels_

# ---------- A. core-cluster sensitivity ----------
lab42 = cluster(42)
core_mask = lab42 != -1
print(f"[{time.time()-t0:5.0f}s] seed42 reproduced: {core_mask.sum()} core docs "
      f"({100*core_mask.mean():.0f}%), {len(set(lab42))-1} clusters")
# consistency with saved run: core docs should carry the same topic ids
same = (dt.loc[core_mask, "topic"].values == lab42[core_mask]).mean()
print(f"  agreement of reproduced core labels with saved topics: {same:.3f}")

dt["core"] = core_mask
dt["topic_m"] = dt["topic"].replace({29: 18})           # same merge as 05
dth = dt[dt["topic_m"] != 6]
core = dth[dth["core"]]

rep = ["CORE-CLUSTER SENSITIVITY (core = direct HDBSCAN members, seed 42)",
       f"core docs: {core_mask.sum()} of {len(dt)} ({100*core_mask.mean():.1f}%)",
       f"reproduced-vs-saved core label agreement: {same:.3f}", ""]

# RQ1 metrics, full vs core
def rq1(d):
    v = d.dropna(subset=["primary"])
    v = v[v["primary"].str.strip() != ""]
    ct = pd.crosstab(v["topic_m"], v["primary"])
    chi2, p, _, _ = chi2_contingency(ct)
    cv = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    return cv, normalized_mutual_info_score(v["topic_m"], v["primary"])
cv_f, nmi_f = rq1(dth); cv_c, nmi_c = rq1(core)
rep += [f"RQ1 vs Primary Problem: full CramersV={cv_f:.3f} NMI={nmi_f:.3f} | "
        f"core CramersV={cv_c:.3f} NMI={nmi_c:.3f}", ""]

# RQ4 auto-detect share per topic, full vs core
def auto_share(d):
    e = d[["acn", "topic_m", "detector"]].dropna()
    e["det"] = e["detector"].str.split(r"\s*;\s*")
    e = e.explode("det").reset_index(drop=True)
    e["auto"] = e["det"].str.contains("Automation", case=False, na=False)
    per = e.drop_duplicates(["acn", "auto"]).groupby("topic_m")["auto"].mean()
    return per
a_f, a_c = auto_share(dth), auto_share(core)
both = pd.DataFrame({"full": a_f, "core": a_c}).dropna()
rho, _ = spearmanr(both["full"], both["core"])
rep += [f"RQ4 automation-detection share: Spearman rho full-vs-core = {rho:.3f} "
        f"({len(both)} themes)",
        f"  taxi/hold-short theme (id 0): full={both.loc[0,'full']:.3f} "
        f"core={both.loc[0,'core']:.3f}",
        f"  TCAS theme (id 18): full={both.loc[18,'full']:.3f} "
        f"core={both.loc[18,'core']:.3f}", ""]
both.to_csv(os.path.join(OUT, "auto_share_full_vs_core.csv"))

# RQ3 slope sign agreement (non-artifact themes)
LEGACY_IDS = {2, 13, 20}
def slopes(d):
    d = d[d["year"].notna()].copy()
    d["year"] = d["year"].astype(int)
    d = d[d["year"] <= 2025]
    tot = d.groupby("year").size()
    pr = d.groupby(["year", "topic_m"]).size().unstack(fill_value=0).div(tot, axis=0)
    return {t: theilslopes(pr[t].values, pr[t].index)[0] for t in pr.columns
            if t not in LEGACY_IDS}
s_f, s_c = slopes(dth), slopes(core)
common = set(s_f) & set(s_c)
agree = np.mean([np.sign(s_f[t]) == np.sign(s_c[t]) for t in common])
rep += [f"RQ3 Theil-Sen slope sign agreement (non-artifact themes): "
        f"{100*agree:.0f}% of {len(common)} themes", ""]

# ---------- B. seed stability ----------
rep += ["SEED STABILITY (UMAP seeds 1,2,3 vs 42; jointly-core docs)"]
stab = []
for s in (1, 2, 3):
    lab_s = cluster(s)
    m = (lab42 != -1) & (lab_s != -1)
    ari = adjusted_rand_score(lab42[m], lab_s[m])
    nmi = normalized_mutual_info_score(lab42[m], lab_s[m])
    ncl = len(set(lab_s)) - 1
    stab.append({"seed": s, "clusters": ncl, "ARI": round(ari, 3),
                 "NMI": round(nmi, 3), "joint_core_n": int(m.sum())})
    rep.append(f"  seed {s}: {ncl} clusters, ARI={ari:.3f}, NMI={nmi:.3f} "
               f"(n={m.sum()})")
    print(f"[{time.time()-t0:5.0f}s] seed {s} done")
pd.DataFrame(stab).to_csv(os.path.join(OUT, "seed_stability.csv"), index=False)
rep.append("")

# ---------- B2. coherence ----------
from bertopic import BERTopic
tm = BERTopic.load(os.path.join(TM, "bertopic_model"))
topics_words = {t: [w for w, _ in tm.get_topic(t)][:10]
                for t in sorted(set(dt["topic_m"])) if t != -1 and tm.get_topic(t)}
tokenized = [d.split() for d in docs]
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
dic = Dictionary(tokenized)
print(f"[{time.time()-t0:5.0f}s] dictionary built ({len(dic)} terms)")
topic_lists = [w for w in topics_words.values()]
for metric in ("c_v", "c_npmi"):
    cm = CoherenceModel(topics=topic_lists, texts=tokenized, dictionary=dic,
                        coherence=metric, processes=1)
    score = cm.get_coherence()
    rep.append(f"COHERENCE {metric}: {score:.3f} "
               f"(mean over {len(topic_lists)} topics, top-10 words)")
    print(f"[{time.time()-t0:5.0f}s] {metric} = {score:.3f}")

with open(os.path.join(OUT, "sensitivity_report.txt"), "w") as fh:
    fh.write("\n".join(rep) + "\n")
print("\n".join(rep))
print("saved to", OUT)
