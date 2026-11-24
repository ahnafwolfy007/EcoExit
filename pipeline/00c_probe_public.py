#!/usr/bin/env python
"""Are the missing files the images of people? (metadata and HEAD requests only)

For each candidate corpus: the share of reachable files among a random 600
images labelled human or vehicle, 600 labelled unknown, and 600 others. Prints
counts by those three groups only.
"""
import importlib.util
import os
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sunsched.cli import base_parser
from sunsched.data import lila

spec = importlib.util.spec_from_file_location("elig", os.path.join(os.path.dirname(os.path.abspath(__file__)), "00b_verify_eligibility.py"))
elig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(elig)

PATTERNS = {**lila.DATASETS, "lila_lynx": r"wsu lynx"}


def main():
    args = base_parser(__doc__).parse_args()
    path = f"{args.data}/lila_image_urls_and_labels.csv.zip"
    if not os.path.exists(path):
        os.makedirs(args.data, exist_ok=True)
        import urllib.request
        req = urllib.request.Request(lila.CSV_URL, headers={"User-Agent": "SunSched/2.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
            while (b := r.read(1 << 22)):
                f.write(b)
    z = zipfile.ZipFile(path)
    inner = [n for n in z.namelist() if n.endswith(".csv")][0]
    rng = np.random.default_rng(0)
    keep = {}
    use = ["dataset_name", "url_gcp", "url_aws", "original_label", "common_name"]
    for ch in pd.read_csv(z.open(inner), usecols=use, dtype=str, chunksize=1_000_000, keep_default_na=False):
        low = ch.dataset_name.str.lower()
        for c, pat in PATTERNS.items():
            g = ch[low.str.contains(pat, regex=True)]
            if not len(g):
                continue
            lab = [lila.simple_label(a, b) for a, b in zip(g.original_label.str.lower().str.strip(), g.common_name.str.lower().str.strip())]
            kind = np.array(["person" if x in ("human", "vehicle") else "unknown" if x == "unknown" else "other" for x in lab])
            key = rng.random(len(g))
            for k in ("person", "unknown", "other"):
                m = kind == k
                part = pd.DataFrame(dict(key=key[m], a=g.url_gcp.values[m], b=g.url_aws.values[m]))
                part = part.nsmallest(600, "key")
                old = keep.get((c, k))
                keep[(c, k)] = (part if old is None else pd.concat([old[0], part]).nsmallest(600, "key"), int(m.sum()) + (0 if old is None else old[1]))
    rows = []
    with ThreadPoolExecutor(12) as ex:
        for (c, k), (d, n) in sorted(keep.items()):
            st = Counter(ex.map(elig.head_status, d.a))
            ok = st.get("200", 0) / max(len(d), 1)
            rows.append(dict(corpus=c, group=k, rows=n, probed=len(d), reachable=ok, status=dict(st)))
            print(f"  {c:18} {k:8} rows {n:9,d}  probed {len(d):4d}  reachable {ok:.4f}  {dict(st)}", flush=True)
    os.makedirs(f"{args.out}/eligibility", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{args.out}/eligibility/public_probe.csv", index=False)


if __name__ == "__main__":
    main()
