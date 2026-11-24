#!/usr/bin/env python
"""WP1: metadata-only eligibility of the SolCascade test corpora (Kaggle, CPU).

    python pipeline/00b_verify_eligibility.py --data /tmp/data --out outputs

Reads LILA's per-image table. Prints and saves counts and pass/fail only (see
the firewall note in sunsched/data/eligibility.py). Image URLs are probed with
HEAD requests; no image is downloaded.
"""
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sunsched.cli import banner, base_parser
from sunsched.data import eligibility as E

CSV_URL = "https://lila.science/public/lila_image_urls_and_labels.csv.zip"
# id -> (dataset name pattern, location pattern that must match, location pattern that must not)
CORPORA = {
    "T1_swg": (r"swg", None, None),
    "T2_orinoquia": (r"orinoqu", None, None),
    "T3_ammonitor": (r"ammonitor", None, None),
    "T4_safari_south": (r"snapshot safari 2024", r"APN|KRU|CDB|KAR|PLN|TSW|NIA", r"SER|ENO|RUA"),
    "T5_wellington": (r"wellington", None, None),
    "B1_island": (r"island conservation", None, None),
    "B2_nz": (r"new zealand|nz[- ]trail", None, None),
    "B3_idaho": (r"idaho", None, None),
    "B4_wcs": (r"wcs", None, None),
    "B6_channel": (r"channel islands", None, None),
    # Added after T4 was found to have no capture times. Africa first, then others.
    "B7_nkhotakota": (r"nkhotakota", None, None),
    "B8_south_africa": (r"snapshot (?:camdeboo|karoo|kgalagadi|kruger|mountain zebra)", None, None),
    "B9_wsu_lynx": (r"wsu lynx", None, None),
    "B10_felidae": (r"felidae", None, None),
}


def pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


def head_status(url):
    """HTTP status of a HEAD request, or the error name. File names contain
    spaces and other characters that must be quoted (the first run of this
    script did not, and counted those images as missing)."""
    u = urllib.parse.quote(url, safe=":/%?=&+,;@~!$'()*")
    last = "error"
    for k in range(4):
        try:
            r = urllib.request.Request(u, method="HEAD", headers={"User-Agent": "SunSched/2.0"})
            with urllib.request.urlopen(r, timeout=30) as f:
                return str(f.status)
        except urllib.error.HTTPError as e:
            last = str(e.code)
            if e.code in (403, 404, 410):
                return last
        except Exception as e:
            last = type(e).__name__
        time.sleep(2.0 * (k + 1))
    return last


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--probe", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(args.data, exist_ok=True)
    outdir = f"{args.out}/eligibility"
    os.makedirs(outdir, exist_ok=True)
    path = f"{args.data}/lila_image_urls_and_labels.csv.zip"
    banner("1. LILA per-image table")
    if not os.path.exists(path):
        req = urllib.request.Request(CSV_URL, headers={"User-Agent": "SunSched/2.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
    print(f"  {os.path.getsize(path) / 1e9:.2f} GB")
    z = zipfile.ZipFile(path)
    inner = [n for n in z.namelist() if n.endswith(".csv")][0]
    cols = pd.read_csv(z.open(inner), nrows=0).columns.tolist()
    print("  columns:", cols)
    c_ds = pick(cols, "dataset_name", "dataset")
    c_id = pick(cols, "image_id", "id")
    c_loc = pick(cols, "location_id", "location")
    c_dt = pick(cols, "datetime", "date_captured", "timestamp")
    c_seq = pick(cols, "sequence_id", "seq_id")
    c_lab = pick(cols, "original_label", "scientific_name", "common_name")
    c_com = pick(cols, "common_name")
    c_url = pick(cols, "url_gcp", "url", "url_aws", "url_azure")
    c_alt = pick(cols, "url_aws", "url_azure")
    need = dict(dataset=c_ds, location=c_loc, datetime=c_dt, label=c_lab, url=c_url, image=c_id)
    print("  mapped:", {**need, "sequence": c_seq})
    if None in (c_ds, c_loc, c_lab):
        sys.exit("cannot map the columns; nothing further was read")
    use = [c for c in dict.fromkeys([c_ds, c_id, c_loc, c_dt, c_seq, c_lab, c_com, c_url, c_alt]) if c]
    parts, seen = {k: [] for k in CORPORA}, {}
    for ch in pd.read_csv(z.open(inner), usecols=use, dtype=str, chunksize=1_000_000, keep_default_na=False):
        for n, k in ch[c_ds].value_counts().items():
            seen[n] = seen.get(n, 0) + int(k)
        low = ch[c_ds].str.lower()
        for cid, (ds, inc, exc) in CORPORA.items():
            m = low.str.contains(ds, regex=True)
            if inc:
                m &= ch[c_loc].str.contains(inc) & ~ch[c_loc].str.contains(exc)
            if m.any():
                g = ch[m]
                parts[cid].append(pd.DataFrame(dict(
                    image=g[c_id].values if c_id else g.index.values, location=g[c_loc].values,
                    dt=g[c_dt].values if c_dt else "", seq=(g[c_seq].values if c_seq else ""),
                    label=g[c_lab].str.lower().str.strip().values,
                    common=(g[c_com].str.lower().str.strip().values if c_com else ""),
                    url=g[c_url].values if c_url else "", alt=g[c_alt].values if c_alt else "")))
    banner("2. datasets in the table")
    for n, k in sorted(seen.items()):
        print(f"  {k:9,d}  {n}")

    banner("3. gates")
    rows, splits = [], {}
    for cid in CORPORA:
        if not parts[cid]:
            print(f"  {cid}: not in the table")
            rows.append(dict(corpus=cid, in_table=False))
            continue
        d = pd.concat(parts[cid], ignore_index=True)
        parts[cid] = None
        d["empty"] = d.label.isin(E.EMPTY) | d.common.isin(E.EMPTY)
        d["animal"] = ~(d.label.isin(E.NON_ANIMAL) | d.common.isin(E.NON_ANIMAL))
        d = d.sort_values("animal", ascending=False).drop_duplicates("image")
        d["t"] = E.parse_times(d.dt)
        has_seq = float(d.seq.astype(str).str.len().gt(0).mean())
        rng = np.random.default_rng(0)
        urls = d.url[d.url.astype(str).str.startswith("http")]
        pick_rows = urls.index[rng.permutation(len(urls))[:args.probe]]
        with ThreadPoolExecutor(8) as ex:
            first = list(ex.map(head_status, d.url.loc[pick_rows]))
            again = [i for i, x in zip(pick_rows, first) if x != "200"]
            second = dict(zip(again, ex.map(head_status, d.alt.loc[again])))
        good = [x == "200" or second.get(i) == "200" for i, x in zip(pick_rows, first)]
        ok = float(np.mean(good)) if good else 0.0
        print(f"  {cid:16} URL status {dict(Counter(first))}; mirror for the rest {dict(Counter(second.values()))}", flush=True)
        for days in (60, 30):                     # 30 days: the plan's risk R2, reported, not used for the gate
            g, cams, roles = E.gates(d[["location", "t", "empty", "label", "animal"]], has_seq, ok, min_days=days)
            rows.append(dict(corpus=cid, in_table=True, min_days=days, **g))
            if days == 60:
                splits[cid] = roles
                cams.drop(columns=["empty"]).to_csv(f"{outdir}/cameras_{cid}.csv", index=False)
            print(f"  {cid:16} {days} d: " + ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in g.items()), flush=True)
    pd.DataFrame(rows).to_csv(f"{outdir}/eligibility.csv", index=False)
    with open(f"{outdir}/splits_60d.json", "w") as f:
        json.dump(splits, f)
    blob = open(f"{outdir}/splits_60d.json", "rb").read()
    print("  splits sha256:", hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    main()
