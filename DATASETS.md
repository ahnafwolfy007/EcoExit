# Datasets

Everything SunSched needs, with direct links. There are two camera-trap corpora,
with different roles (see [PREREGISTRATION.md](PREREGISTRATION.md)):

| Corpus | Role | Download |
|---|---|---|
| **CCT20** (Caltech Camera Traps) | Development: design, debugging, tuning | ~6.6 GB |
| **Snapshot Serengeti**, season 10 | **Test**: the pre-registered hypotheses are decided here, once | ~17 GB transferred, ~1 GB kept |

The easy way is one command per corpus. It downloads into `./data/` and resumes
if interrupted:

```bash
python pipeline/00_fetch_datasets.py --corpus cct20
python pipeline/00_fetch_datasets.py --corpus serengeti
```

Nothing needs an account, an API key, or a GPU.

---

## 1. Development corpus: CCT20 camera-trap images and labels

**Caltech Camera Traps, CCT20 benchmark subset** (Beery, Van Horn and Perona,
"Recognition in Terra Incognita", ECCV 2018). Dataset page and licence:
https://lila.science/datasets/caltech-camera-traps

| File | Link | Size | Goes to |
|---|---|---|---|
| Images (downsized to at most 1024 px) | https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz | 6,492,615,601 bytes (6.49 GB) | `data/cct20/eccv_18_all_images_sm.tar.gz` |
| Annotations and official splits | https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz | 2,997,071 bytes (3 MB) | `data/cct20/`, then extract |

**Do not extract the image archive.** The pipeline streams the images directly out
of the `.tar.gz`, which saves 6.5 GB of disk. Extract only the annotations archive,
which gives `data/cct20/eccv_18_annotation_files/` with five JSON files.

What is in it: 57,864 images from 20 camera locations in the American Southwest,
16 classes (15 animal/vehicle classes plus `empty`), and for every image its real
capture time, camera location and burst ID. The official splits are
**location-disjoint**, and the pipeline uses them as follows:

| Split | Images | Cameras | Pipeline role |
|---|---|---|---|
| train, cis_val, cis_test | 32,864 | 10 | `train`: fitting the exit heads and the detector |
| trans_val | 1,725 | 1 (new) | `calib`: calibrating confidence and refinement gains |
| trans_test | 23,275 | 9 (new) | `eval`: the simulated deployments |

No simulated camera is ever seen in training. Never replace these with a random
split: a camera trap's background never changes, so a random split rewards a model
for memorising backgrounds.

The 9 `trans_test` cameras were already used by the first full run. That is why
CCT20 is now the **development** corpus, where tuning is allowed, and the claims
are tested on Serengeti instead.

Manual download, if the script is blocked (curl resumes with `-C -`):

```bash
mkdir -p data/cct20
curl -L -C - -o data/cct20/eccv_18_all_images_sm.tar.gz https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz
curl -L -o data/cct20/eccv_18_annotations.tar.gz https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz
tar -xzf data/cct20/eccv_18_annotations.tar.gz -C data/cct20
```

## 2. Test corpus: Snapshot Serengeti (season 10)

Swanson et al., "Snapshot Serengeti, high-frequency annotated camera trap images of
40 mammalian species in an African savanna", *Scientific Data* 2015. Dataset page:
https://lila.science/datasets/snapshot-serengeti. Licence: Community Data License
Agreement, permissive variant.

It is a different continent, biome and species set from the development data, and
nobody on this project had looked at it when PREREGISTRATION.md was written.

| File | Link | Size |
|---|---|---|
| Season 10 metadata (COCO Camera Traps JSON, zipped) | https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-v-2-0/SnapshotSerengetiS10.json.zip | 33,924,894 bytes (34 MB) |
| Official location-disjoint splits | https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-v-2-0/SnapshotSerengetiSplits_v0.json | 4,734 bytes |
| Individual images (GCP) | `https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-unzipped/<file_name>` | ~0.6 MB each |
| Individual images (Azure mirror) | `https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped/<file_name>` | same |

`<file_name>` is the `file_name` field in the metadata, for example
`S1/B04/B04_R1/S1_B04_R1_PICT0012.JPG` (checked on 2026-09-28: 635,647 bytes, both mirrors).

**Only a subset is downloaded, picked by a fixed rule.** The full corpus is several
terabytes. The fetch script applies the selection written into
`sunsched/data/serengeti.py` before any Serengeti data was seen:

| Role | What | Default size |
|---|---|---|
| train | whole bursts sampled from official `train` locations, 20% empty | 10,000 images |
| eval | official `val` cameras, most animal activity first, every image, until 15,000 images | ~15,000 images |
| calib | the next 3 `val` cameras in that ranking | a few thousand images |

Each image is fetched once, downscaled to 320 px high on arrival, and cached in
`data/serengeti/images_small/` at about 25 KB. That is roughly **17 GB
transferred and 1 GB kept**, and the full-size originals are never stored.
Re-running the script skips images already cached. The `--quick` preset selects
about 5,000 images instead.

There are no manual download commands for Serengeti images: they come one by one.
Use the script.

## 3. Solar irradiance and air temperature (required)

**PVGIS**, the EU Joint Research Centre's solar-radiation service. It gives hourly
irradiance on the panel plane and 2 m air temperature, with no signup.
Tool: https://re.jrc.ec.europa.eu/pvg_tools/en/

The pipeline calls the API for each site for the years 2011–2015 and caches the
replies in `data/pvgis/` (about 2 MB per site). Example request for the main site:

https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=33.0&lon=-116.8&startyear=2011&endyear=2015&pvcalculation=0&angle=33&aspect=0&outputformat=json

| Site | Lat, lon | Role |
|---|---|---|
| `cct_region` | 33.0, −116.8 | Main site for CCT20 (Southern California). Region-level only: the dataset does not publish camera coordinates. |
| `serengeti` | −2.33, 34.83 | Main site for Snapshot Serengeti (central Serengeti National Park) |
| `phoenix`, `nairobi`, `dhaka`, `munich`, `bergen` | see `sunsched/config.py` | Stress tests: the same real animal activity under a different sun |

If PVGIS cannot be reached, `--solar analytic` runs the pipeline on a built-in
synthetic weather generator. The results are then labelled synthetic throughout and
must not be reported as real-weather results.

## 4. Pretrained network weights (required)

**MobileNetV3-Large, ImageNet weights (torchvision IMAGENET1K_V2)**, about 22 MB.
torchvision downloads it automatically on first use:
https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth

Nothing else is needed. The pipeline uses only the files listed above.

---

## Where everything ends up

```
data/
  cct20/
    eccv_18_all_images_sm.tar.gz      6.49 GB, keep compressed
    eccv_18_annotations.tar.gz
    eccv_18_annotation_files/         5 split JSON files
  serengeti/
    SnapshotSerengetiS10.json.zip     season metadata
    SnapshotSerengetiSplits_v0.json   official splits
    images_small/                     downscaled cache of the selected images (~1 GB)
  pvgis/                              cached PVGIS replies, one per site
  fetch_manifest_<corpus>.json        what the fetch script got
```

`data/` is gitignored and never committed.

## Sources the model constants come from

These are not downloads, but the results depend on them:

- Snapshot Serengeti: A. Swanson et al., "Snapshot Serengeti, high-frequency
  annotated camera trap images of 40 mammalian species in an African savanna",
  *Scientific Data* 2, 150026 (2015).
- CCT20: S. Beery, G. Van Horn, P. Perona, "Recognition in Terra Incognita",
  ECCV 2018.

- Battery ageing: B. Xu, A. Oudalov, A. Ulbig, G. Andersson, D. Kirschen, "Modeling of
  Lithium-Ion Battery Degradation for Cell Life Assessment", *IEEE Trans. Smart Grid*,
  2018. **Check the constants in `AgingCfg` against the paper's parameter table
  before reporting any number.** They are also swept ±50%.
- Risk-controlled reserve: I. Gibbs and E. Candès, "Adaptive Conformal Inference Under
  Distribution Shift", NeurIPS 2021.
- Node power figures (`NodeCfg`) are datasheet-typical assumptions. There is no
  hardware in this project, so every one of them is swept.
