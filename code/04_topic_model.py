"""
04_topic_model.py
BERTopic topic modeling of the ATC-related ASRS narratives (RQ1 input).

Pipeline
  1. Load merged corpus (30,410 reports), light ASRS-specific text cleaning.
  2. Sentence embeddings: all-MiniLM-L6-v2 (cached to .npy — re-runs are fast).
  3. BERTopic: UMAP -> HDBSCAN -> c-TF-IDF, fixed random_state for
     reproducibility, min_topic_size sized for a ~30k corpus.
  4. Outputs (results/topic_model/):
       topic_info.csv          one row per topic: size, top words, name
       doc_topics.csv          ACN, year, topic id, probability + the structured
                               fields needed for RQ1 validation and RQ2-RQ4
                               cross-analysis (Anomaly, Primary Problem,
                               Detector, Result, Flight Phase, Function)
       topics_top_words.txt    human-readable topic summaries
       bertopic_model/         saved model (safetensors)
       embeddings.npy          cached embeddings

Run:  python 04_topic_model.py
Notes: first run downloads the embedding model (~90 MB) and embeds 30k docs on
CPU (several minutes). Subsequent runs reuse embeddings.npy.
"""
import os, re, sys, time
import numpy as np
import pandas as pd

# Repository root. Override with the ASRS_BASE environment variable, e.g.
#   set ASRS_BASE=D:\my\data\dir      (Windows)
#   export ASRS_BASE=/home/me/data     (Linux/macOS)
# Default: the parent of this code/ directory, i.e. the repository root.
BASE = os.environ.get(
    "ASRS_BASE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
IN     = os.path.join(BASE, "Dataset", "merged", "asrs_merged_dedup.csv")
OUT    = os.path.join(BASE, "results", "topic_model")
os.makedirs(OUT, exist_ok=True)

SEED           = 42
EMB_MODEL      = "all-MiniLM-L6-v2"
MIN_TOPIC_SIZE = 100          # ~0.3% of corpus; keeps topics interpretable
N_NEIGHBORS    = 15
N_COMPONENTS   = 5

t0 = time.time()
df = pd.read_csv(IN, dtype=str, low_memory=False)
print(f"[{time.time()-t0:6.1f}s] loaded {len(df)} reports")

def find(*kw):
    for c in df.columns:
        if all(k.lower() in c.lower() for k in kw):
            return c
    return None

COLS = {
    "acn":       "ACN",
    "narrative": find("Report 1", "Narrative"),
    "synopsis":  find("Report 1", "Synopsis"),
    "anomaly":   find("Events", "Anomaly"),
    "primary":   find("Assessments", "Primary Problem"),
    "detector":  find("Events", "Detector"),
    "when":      find("Events", "When Detected"),
    "result":    find("Events", "Result"),
    "phase":     find("Aircraft 1", "Flight Phase"),
    "function":  find("Person 1", "Function"),
    "hf":        find("Person 1", "Human Factors"),
    "date":      find("Date") or df.columns[1],
}
df["_year"] = pd.to_numeric(
    df[COLS["date"]].fillna("").astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce")

# ---- text cleaning (ASRS-specific, light-touch: embeddings want real text) ----
# Pre-~2009 ASRS narratives use heavy abbreviation (tfc, ctlr, clred, ...);
# later ones use full words. Without normalization the model splits the SAME
# hazard into era-duplicate topics (observed in run 1: txwy vs taxiway,
# tkof clrnc vs takeoff clearance) and RQ3 trends would be confounded by
# writing style. Expansions below follow the official ASRS decode list.
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

def clean(t: str) -> str:
    t = str(t)
    # pre-~2009 reports are ALL CAPS -> lowercase everything for consistency
    t = t.lower()
    # de-identification artifacts and report boilerplate
    t = re.sub(r"\bcallback\b.*?$", " ", t)
    t = re.sub(r"\[.*?\]", " ", t)              # bracketed redactions
    t = re.sub(r"\b[a-z]{3}\d*\.(airport|tracon|artcc|vor)\b", " ", t)  # coded facilities
    t = re.sub(r"\bzzz\w*\b", " ", t)           # de-identified place tokens
    t = _abbrev_re.sub(lambda m: ABBREV[m.group(1)], t)
    t = re.sub(r"\bfl\d{3}\b", " flight level ", t)  # FL310 etc.
    t = re.sub(r"\s+", " ", t).strip()
    return t

texts = (df[COLS["narrative"]].fillna("")).map(clean).tolist()
keep = [i for i, t in enumerate(texts) if len(t.split()) >= 20]
print(f"[{time.time()-t0:6.1f}s] cleaned; {len(keep)} docs with >=20 words")
df_k  = df.iloc[keep].reset_index(drop=True)
docs  = [texts[i] for i in keep]

# ---- embeddings (cached) ----
emb_path = os.path.join(OUT, "embeddings.npy")
from sentence_transformers import SentenceTransformer
if os.path.exists(emb_path):
    embeddings = np.load(emb_path)
    assert embeddings.shape[0] == len(docs), "cache mismatch — delete embeddings.npy"
    print(f"[{time.time()-t0:6.1f}s] loaded cached embeddings {embeddings.shape}")
else:
    st = SentenceTransformer(EMB_MODEL)
    embeddings = st.encode(docs, batch_size=64, show_progress_bar=True,
                           convert_to_numpy=True)
    np.save(emb_path, embeddings)
    print(f"[{time.time()-t0:6.1f}s] embedded {embeddings.shape}")

# ---- BERTopic ----
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

umap_model = UMAP(n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS,
                  min_dist=0.0, metric="cosine", random_state=SEED)
# "leaf" selection is essential here: the corpus is homogeneous by construction
# (all ATC-related), and "eom" collapses it into 2 giant reporter-style blobs
# (~18k pilot vs ~12k controller narratives). Leaf exposes the substructure
# (~35 topics at mcs=100 per the parameter sweep); the resulting outliers are
# reassigned to topics via c-TF-IDF below.
hdbscan_model = HDBSCAN(min_cluster_size=MIN_TOPIC_SIZE, min_samples=10,
                        metric="euclidean",
                        cluster_selection_method="leaf", prediction_data=True)
# domain stopwords: generic aviation/report words that dominate every topic
DOMAIN_STOP = ["aircraft", "acft", "flight", "pilot", "captain", "atc", "controller",
               "reported", "reporter", "stated", "advised", "time", "airport", "apt",
               "runway", "rwy", "approach", "apch", "altitude", "alt", "feet", "ft",
               "just", "did", "told", "said", "asked"]
from sklearn.feature_extraction import text as sk_text
stop = list(sk_text.ENGLISH_STOP_WORDS.union(DOMAIN_STOP))
# NOTE: BERTopic's c-TF-IDF sees ONE concatenated document per topic, so
# min_df must stay tiny (min_df=10 crashed when topics < 10)
vectorizer = CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=2)

topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                       vectorizer_model=vectorizer, calculate_probabilities=False,
                       verbose=True)
topics, probs = topic_model.fit_transform(docs, embeddings)
print(f"[{time.time()-t0:6.1f}s] fitted: {len(set(topics)) - (1 if -1 in topics else 0)} topics "
      f"(+outliers: {sum(1 for t in topics if t == -1)})")

# reduce outliers into nearest topics (keeps corpus fully assigned for RQ2-RQ4)
if -1 in topics:
    topics = topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf")
    topic_model.update_topics(docs, topics=topics, vectorizer_model=vectorizer)
    print(f"[{time.time()-t0:6.1f}s] outliers reassigned")

# ---- outputs ----
info = topic_model.get_topic_info()
info.to_csv(os.path.join(OUT, "topic_info.csv"), index=False)

out_cols = {k: COLS[k] for k in
            ["acn", "anomaly", "primary", "detector", "when", "result",
             "phase", "function", "hf"]}
doc_out = df_k[[c for c in out_cols.values()]].copy()
doc_out.columns = list(out_cols.keys())
doc_out["year"]  = df_k["_year"]
doc_out["topic"] = topics
doc_out.to_csv(os.path.join(OUT, "doc_topics.csv"), index=False)

with open(os.path.join(OUT, "topics_top_words.txt"), "w", encoding="utf-8") as fh:
    for tid in sorted(set(topics)):
        words = [w for w, _ in topic_model.get_topic(tid)][:12]
        size = sum(1 for t in topics if t == tid)
        fh.write(f"Topic {tid:3d}  (n={size:5d}): {', '.join(words)}\n")

topic_model.save(os.path.join(OUT, "bertopic_model"), serialization="safetensors",
                 save_ctfidf=True, save_embedding_model=EMB_MODEL)
print(f"[{time.time()-t0:6.1f}s] all outputs saved to {OUT}")
