# Datasets

Everything the pipeline can use, with direct links. Every link below was
fetched and returned HTTP 200 on 23 Sep 2026 unless marked otherwise; sizes are
the actual `Content-Length` returned.

**You do not need all of this.** The quick start needs one 170 MB download
(CIFAR-100) and nothing else. The tiers below say what each one buys you.

```bash
python scripts/00_fetch_datasets.py          # tier 1 + 2 (~180 MB)
python scripts/00_fetch_datasets.py --all    # adds Serengeti (~370 MB total)
```

---

## Tier 1 — required (the pipeline will not run without it)

| Dataset | Link | Size | Purpose |
|---|---|---|---|
| CIFAR-100 | https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz | 169 MB | Images the backbone is trained and scored on. Downloaded automatically by torchvision; the direct link is only for a manual fallback. |

Extract a manual download into `./data/` so you end up with
`./data/cifar-100-python/`.

---

## Tier 2 — real event timing (removes the biggest reviewer objection)

The single most valuable thing in this list. Camera-trap `.json` metadata is
COCO-format and carries **real capture timestamps, real camera locations and
real burst structure** — so it gives a genuine event-arrival process for a
9 MB download, with **no images and no GPU**. This is what replaces the
synthetic Poisson event generator.

Verified in the Caltech metadata: **243,100 records, 100% with parseable
timestamps, 140 locations, 51.7% empty frames.**

| Dataset | Link | Size | Purpose |
|---|---|---|---|
| Caltech Camera Traps metadata | https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip | 9.0 MB | Primary event-arrival process. |
| Caltech splits | https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/CaltechCameraTrapsSplits_v0.json | < 0.1 MB | Official location-disjoint splits. |
| Snapshot Serengeti metadata | https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-v-2-0/SnapshotSerengeti_S1-11_v2_1.json.zip | 188.5 MB | Scale, plus 11 seasons for the temporal-shift split. |
| Serengeti splits | https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-v-2-0/SnapshotSerengetiSplits_v0.json | < 0.1 MB | Official splits: 179 train / 46 val locations, zero overlap (verified). |
| LILA index (all camera-trap corpora) | https://lila.science/datasets/ | — | Browse NACTI, WCS, and others. |
| iWildCam competition data | https://github.com/visipedia/iwildcam_comp | — | Competition metadata and baselines. |
| WILDS (iWildCam-WILDS) | https://wilds.stanford.edu/datasets/ | — | Established out-of-distribution leaderboard. |

**Two traps, both of which silently destroy results:**

1. **Never generate your own random split.** A camera trap is bolted to a tree
   and never moves, so every frame from a location shares a pixel-identical
   background. On a random split a model learns the background, infers the
   location, and predicts from that location's species prior — scoring well
   without ever looking at an animal, then collapsing on a new camera. Use the
   official location-disjoint split files above.

2. **Watch the bursts.** Traps fire in bursts of near-identical frames
   fractions of a second apart. If one frame of a burst is in train and another
   in test, that is memorization measured as generalization. `seq_id` in the
   metadata identifies the burst; `ecoexit/data/camera_traps.py`
   (`assert_no_sequence_leakage`) checks it and the check is reported.

**Data-hygiene note:** a small number of Caltech timestamps are malformed (the
minimum sorts as the literal string `0000-00-00 00:00:00`). `parse_timestamp`
filters on parse success — without that, every malformed capture lands in the
same slot at the epoch.

---

## Tier 3 — real irradiance

| Source | Link | Coverage | Status |
|---|---|---|---|
| **PVGIS** (used by default) | https://re.jrc.ec.europa.eu/pvg_tools/en/ | Europe, Africa, Asia, Americas; hourly | Verified live: returned 8,760 hourly records for the Serengeti coordinates. No signup, no API key. |
| SURFRAD | https://gml.noaa.gov/grad/surfrad/ | 7 US stations, 1-minute | 200 |
| BSRN | https://bsrn.awi.de/ | Global research-grade network | 200 |
| DKASC Alice Springs | https://dkasolarcentre.com.au/download | Real PV output, 1-minute | 200 |
| NREL NSRDB | https://nsrdb.nrel.gov/ | Global satellite, 5–30 min | Did not resolve from the machine used to verify; retry from yours. PVGIS is the working substitute. |

PVGIS is fetched and cached automatically by
`scripts/00_fetch_datasets.py --only pvgis`. Pass `--source pvgis` to
`scripts/03_build_traces.py` to use it instead of the analytic generator.

The analytic Haurwitz-plus-AR(1) generator stays as the offline fallback, so
the repository still runs with no network at all.

---

## Tier 4 — measured battery degradation

Needed to replace the literature cycle-life curve in `BatteryCfg` with measured
data, and to run Claim 1 across chemistries. None of these are auto-downloaded;
the pipeline uses the datasheet curve until you point it at real cells.

| Source | Link | Notes |
|---|---|---|
| Severson et al. (Stanford/Toyota) | https://data.matr.io/1/ | 124 LFP cells cycled to failure. The standard degradation benchmark. |
| NASA PCoE | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ | Several battery-ageing sets. |
| Oxford degradation dataset | https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac | Returned 403 to an automated request; expected to open in a browser. |

To use one: fit `CycleLifeCurve` to the measured depth-vs-cycles points and
replace `BatteryCfg.dod_points` / `cycles_at_dod`. Everything downstream —
the wear price, the carbon accounting — follows automatically.

---

## Tier 5 — backbone grid (Phase 3, needs a GPU)

The bespoke ~100K-parameter network is not comparable to anything published.
Replacing it is Phase 3 work.

| Source | Link | Notes |
|---|---|---|
| `timm` (pretrained backbones) | https://github.com/huggingface/pytorch-image-models | ResNet, MobileNetV2/V3, EfficientNet, RegNet, MobileViT, EfficientViT. |
| MCUNet / MCUNetV2 | https://github.com/mit-han-lab/mcunet | Required if the paper claims MCU-class nodes. |
| MSDNet | https://github.com/gaohuang/MSDNet | Official. Reviewers will insist on this one. |
| RANet | https://github.com/yangle15/RANet-pytorch | Official; resolution-adaptive. |
| Once-for-All | https://github.com/mit-han-lab/once-for-all | Official; elastic subnetworks. |

Use official implementations, not reimplementations — reimplemented baselines
are the most common reason systems papers get accused of strawmanning.

---

## What lands where

```
data/
  cifar-100-python/                       CIFAR-100
  camera_traps/
    caltech_camera_traps.json.zip         as downloaded, 9 MB
    caltech_images_20210113.json          as extracted, 121 MB
    CaltechCameraTrapsSplits_v0.json      official location-disjoint splits
    SnapshotSerengeti_S1-11_v2_1.json.zip (only with --all)
    SnapshotSerengetiSplits_v0.json
  pvgis/
    pvgis_<lat>_<lon>_<year>.json         cached hourly irradiance, one per site
  fetch_manifest.json                     what succeeded, written by scripts/00
```

Note the archives do not extract to the name they are downloaded as — the
Caltech zip unpacks to `caltech_images_20210113.json`. Nothing in the pipeline
hardcodes these names; `ecoexit.data.camera_traps.find_metadata` locates the
metadata by content, so a re-release with a new version stamp still works.

`data/` is gitignored in full. Everything in it is re-fetchable.

## If a download fails

Nothing here needs an account or a key, so a failure is almost always a
firewall or a transient server issue.

- **CIFAR-100 stalls**: it is a ~170 MB pull from a university server that is
  genuinely slow sometimes. Give it a few minutes before assuming it hung.
- **LILA links blocked**: run with `--events synthetic`. The pipeline works end
  to end on the synthetic arrival process; you lose Claim 4 only.
- **PVGIS unreachable**: leave `SolarCfg.source = "analytic"`. The pipeline
  works end to end offline; you lose the real-irradiance claim only.

The pipeline never silently substitutes synthetic data for real data inside a
run labelled as real — `fetch_pvgis_hourly` raises rather than falling back,
because a silent fallback inside a results table is how invalid numbers get
published.
