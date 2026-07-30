# Detection gaps in the pilot–controller system

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21709021.svg)](https://doi.org/10.5281/zenodo.21709021)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproduction code for:

> Jayaraman, D. *Detection gaps in the pilot–controller system: a four-decade
> analysis of air-traffic-control-related incident reports from NASA ASRS
> (1988–2026).* Submitted to *Safety Science*.

The study analyses **30,410 unique ATC-related NASA ASRS reports** (January
1988 – January 2026). Hazard themes are recovered from the reporter-written
narratives with BERTopic, validated against the labels NASA analysts assign to
every processed report, and then linked to operational conditions,
proportional trends across four decades, and — the core contribution — the
**detection pathways** recorded in the Detector, When Detected, and Result
fields.

**Headline result.** The airborne-conflict hazard family is declining as a
share of reports (TCAS resolution advisories: −4.4 percentage points per
decade) and is caught by automation in 48.7% of cases. Surface and tower
hazards are rising, are caught by automation in only 2–10% of cases, and carry
the smallest reported miss distances (median 150 ft for taxi conflicts). The
detection gap has moved to the airport surface.

## Getting the data

The ASRS reports are public but must be exported by the user; NASA's terms do
not permit redistribution of the database here.

1. Go to the [ASRS Database Online](https://asrs.arc.nasa.gov/search/database.html).
2. Query the **Anomaly** field for `ATC Issue All Types`.
3. Because the interface caps each export, split the date range into eight
   windows spanning 1988-01 to 2026-01 and export each as a **tab-separated
   `.xls`** file.
4. Place all eight files in `Dataset/` at the repository root.

The exports carry a two-row header (group header, then field name) followed by
a blank row; `01_merge_asrs.py` expects exactly that layout and builds unique
column names from the two rows.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.12 on a CPU is sufficient. No GPU is required.

By default the scripts resolve paths relative to the repository root. To keep
data elsewhere, set `ASRS_BASE`:

```bash
export ASRS_BASE=/path/to/data     # Windows: set ASRS_BASE=D:\path\to\data
```

That directory must then contain `Dataset/`, and `results/` will be written
into it.

## Running the pipeline

Run the scripts in numerical order. Each writes into `results/` and is safe to
re-run.

```bash
python code/01_merge_asrs.py
python code/02_profile_asrs.py
python code/03_descriptives.py
python code/04_topic_model.py
python code/05_validate_topics.py
python code/06_operational_profiles.py
python code/07_trends.py
python code/08_detection.py
python code/09_severity_priority.py
python code/10_sensitivity_coherence.py
```

| # | Script | Purpose | Key outputs |
|---|--------|---------|-------------|
| 01 | `01_merge_asrs.py` | Merge the 8 TSV exports, build unique column names, de-duplicate on ACN (30,906 → 30,410) | `Dataset/merged/asrs_merged_dedup.csv` |
| 02 | `02_profile_asrs.py` | Temporal coverage and field-population profile | console report |
| 03 | `03_descriptives.py` | Data-section tables and figures | `results/descriptives/` |
| 04 | `04_topic_model.py` | Abbreviation normalisation + BERTopic (36 topics) | `results/topic_model/` |
| 05 | `05_validate_topics.py` | **RQ1** — label topics, merge duplicates, convergent validation vs. NASA labels | `results/rq1_validation/` |
| 06 | `06_operational_profiles.py` | **RQ2** — theme × flight phase / reporter role / human factors | `results/rq2_operational/` |
| 07 | `07_trends.py` | **RQ3** — yearly proportions, Kendall τ + Theil–Sen, BH correction | `results/rq3_trends/` |
| 08 | `08_detection.py` | **RQ4** — detector channel, timing, and outcome per theme | `results/rq4_detection/` |
| 09 | `09_severity_priority.py` | **RQ5** — miss-distance and evasive proxies, priority composite | `results/rq5_priority/` |
| 10 | `10_sensitivity_coherence.py` | Coherence (C_v, NPMI), seed stability, core-cluster sensitivity (Appendix A) | `results/rq_sensitivity/` |

### Runtime

Step 04 embeds ~30,000 narratives and takes roughly **45 minutes on a CPU**
the first time. Embeddings are cached to `results/topic_model/embeddings.npy`,
so subsequent runs of 04 take about **3 minutes**. Every other step runs in
under a minute.

## Mapping from outputs to the paper

| Paper item | Produced by |
|---|---|
| Fig. 1 — reports per year | `03_descriptives.py` |
| Fig. 2 — *When Detected* population rate | `03_descriptives.py` |
| Fig. 3 — theme × Anomaly alignment heatmap | `05_validate_topics.py` |
| Fig. 4 — theme × flight phase | `06_operational_profiles.py` |
| Fig. 5 — trend small multiples | `07_trends.py` |
| Fig. 6 — detection-channel mix | `08_detection.py` |
| Fig. 7 — priority scatter | `09_severity_priority.py` |
| Table 1 — proportional trends | `07_trends.py` |
| Cramér's *V* = 0.73 (role segregation) | `06_operational_profiles.py` |
| Appendix A — sensitivity and stability | `10_sensitivity_coherence.py` |

## Reproducibility notes

- All random seeds are fixed at **42** (UMAP, HDBSCAN, and the sampling steps).
- Clustering uses UMAP (5 components, 15 neighbours) and HDBSCAN
  (`min_cluster_size=100`, `min_samples=10`, **leaf** selection). Leaf
  selection is deliberate: because every report in the corpus is ATC-related,
  the default excess-of-mass selection collapses the corpus into two clusters
  that track pilot-versus-controller *writing style* rather than hazard
  content.
- Narratives are lower-cased and ~60 ASRS abbreviations are expanded before
  modelling. This is not cosmetic. ASRS narrative style changed around 2009,
  and without normalisation the model splits identical hazards into
  era-duplicate topics (`txwy hold short` vs. `taxiway hold short`), which
  would make a change in spelling conventions look like a decline in hazard.
- 63% of documents are HDBSCAN outliers and are reassigned to their nearest
  topic by c-TF-IDF similarity. Script 10 re-runs the headline analyses on
  core-cluster documents only; all conclusions hold (Appendix A of the paper).
- Seed stability across three alternative UMAP seeds: ARI 0.91–0.96, NMI 0.97.

## Citation

This release is archived on Zenodo. Version 1.0.0, which produced the results
in the paper, is [10.5281/zenodo.21709021](https://doi.org/10.5281/zenodo.21709021);
[10.5281/zenodo.21709020](https://doi.org/10.5281/zenodo.21709020) always
resolves to the latest version.

```bibtex
@software{jayaraman2026code,
  author  = {Jayaraman, Deepan},
  title   = {Detection gaps in the pilot--controller system: analysis code},
  version = {v1.0.0},
  year    = {2026},
  doi     = {10.5281/zenodo.21709021},
  url     = {https://doi.org/10.5281/zenodo.21709021}
}

@article{jayaraman2026detection,
  author  = {Jayaraman, Deepan},
  title   = {Detection gaps in the pilot--controller system: a four-decade
             analysis of air-traffic-control-related incident reports from
             {NASA} {ASRS} (1988--2026)},
  journal = {Safety Science},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The
underlying ASRS reports are US Government public-domain records, obtained from
the [NASA ASRS Database Online](https://asrs.arc.nasa.gov).
