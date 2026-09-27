# Datasets

Everything SunSched needs, with direct links. The easy way is one command, which
downloads all of it into `./data/` and resumes if interrupted:

```bash
python pipeline/00_fetch_datasets.py
```

Nothing needs an account, an API key, or a GPU. Total download is about **6.6 GB**,
almost all of it the image archive.

---

## 1. Camera-trap images and labels (required)

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

| Split | Images | Cameras | Used for |
|---|---|---|---|
| train, cis_val, cis_test | 32,864 | 10 | training the exit heads |
| trans_val | 1,725 | 1 (new) | calibrating confidence and refinement gains |
| trans_test | 23,275 | 9 (new) | the simulated deployments, and nothing else |

No simulated camera is ever seen in training. Never replace these with a random
split: a camera trap's background never changes, so a random split rewards a model
for memorising backgrounds.

Manual download, if the script is blocked (curl resumes with `-C -`):

```bash
mkdir -p data/cct20
curl -L -C - -o data/cct20/eccv_18_all_images_sm.tar.gz https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz
curl -L -o data/cct20/eccv_18_annotations.tar.gz https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz
tar -xzf data/cct20/eccv_18_annotations.tar.gz -C data/cct20
```

## 2. Solar irradiance and air temperature (required)

**PVGIS**, the EU Joint Research Centre's solar-radiation service. It gives hourly
irradiance on the panel plane and 2 m air temperature, with no signup.
Tool: https://re.jrc.ec.europa.eu/pvg_tools/en/

The pipeline calls the API for each site for the years 2011–2015 and caches the
replies in `data/pvgis/` (about 2 MB per site). Example request for the main site:

https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=33.0&lon=-116.8&startyear=2011&endyear=2015&pvcalculation=0&angle=33&aspect=0&outputformat=json

| Site | Lat, lon | Role |
|---|---|---|
| `cct_region` | 33.0, −116.8 | Main site, near the CCT cameras (Southern California). Region-level only: the dataset does not publish camera coordinates. |
| `phoenix`, `nairobi`, `dhaka`, `munich`, `bergen` | see `sunsched/config.py` | Stress tests: the same real animal activity under a different sun |

If PVGIS cannot be reached, `--solar analytic` runs the pipeline on a built-in
synthetic weather generator. The results are then labelled synthetic throughout and
must not be reported as real-weather results.

## 3. Pretrained network weights (required)

**MobileNetV3-Large, ImageNet weights (torchvision IMAGENET1K_V2)**, about 22 MB.
torchvision downloads it automatically on first use:
https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth

## 4. Optional

| What | Link | Size | Why |
|---|---|---|---|
| Full Caltech Camera Traps metadata (140 cameras, timestamps only) | https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip | 9 MB | Repeat the day/night timing analysis on all 140 cameras (`--only cct_meta`) |
| Full CCT image set | https://lila.science/datasets/caltech-camera-traps | 105 GB | Not needed |

---

## Where everything ends up

```
data/
  cct20/
    eccv_18_all_images_sm.tar.gz      6.49 GB, keep compressed
    eccv_18_annotations.tar.gz
    eccv_18_annotation_files/         5 split JSON files
  pvgis/                              cached PVGIS replies, one per site
  fetch_manifest.json                 what the fetch script got
```

`data/` is gitignored and never committed.

## Sources the model constants come from

These are not downloads, but the results depend on them:

- Battery ageing: B. Xu, A. Oudalov, A. Ulbig, G. Andersson, D. Kirschen, "Modeling of
  Lithium-Ion Battery Degradation for Cell Life Assessment", *IEEE Trans. Smart Grid*,
  2018. **Check the constants in `AgingCfg` against the paper's parameter table
  before reporting any number.** They are also swept ±50%.
- Risk-controlled reserve: I. Gibbs and E. Candès, "Adaptive Conformal Inference Under
  Distribution Shift", NeurIPS 2021.
- Node power figures (`NodeCfg`) are datasheet-typical assumptions. There is no
  hardware in this project, so every one of them is swept.
