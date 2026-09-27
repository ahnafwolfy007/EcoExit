"""Vision pieces with random weights: no download, no dataset."""
import importlib.util
import io
import os
import tarfile
import tempfile

import numpy as np
from PIL import Image

from sunsched.data import cct20
from sunsched.vision.backbone import TapExtractor, input_size, tap_dims, to_batch
from sunsched.vision.cost import trunk_macs_at_taps
from sunsched.vision.heads import fit_temperature, train_head

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAPS = (6, 12, 16)


def _trunk():
    from torchvision.models import mobilenet_v3_large
    return mobilenet_v3_large(weights=None).features.eval()


def test_tap_dimensions_and_macs():
    trunk = _trunk()
    ext = TapExtractor(trunk, TAPS)
    h, w = input_size(96, 1024 / 747)
    assert (h, w) == (96, 128)
    # avg + max + 2x2 grid (6C) for the early taps, avg + max (2C) for the last
    assert tap_dims(ext, h, w) == [40 * 6, 112 * 6, 960 * 2]
    macs = trunk_macs_at_taps(trunk, TAPS, h, w)
    vals = [macs[t] for t in TAPS]
    assert all(v > 0 for v in vals) and vals == sorted(vals)
    big = trunk_macs_at_taps(trunk, TAPS, 2 * h, 2 * w)
    assert 3.0 < big[16] / macs[16] < 5.0, "MACs should scale with pixel count"


def test_heads_learn_and_calibrate():
    rng = np.random.default_rng(0)
    n, d, k = 1200, 20, 4
    y = rng.integers(0, k, n)
    X = rng.normal(0, 3, (k, d))[y] + rng.normal(0, 1, (n, d))
    th = train_head(X[:900], y[:900], k, hidden=32, dropout=0.1, epochs=20, lr=1e-2,
                    weight_decay=1e-4, seed=0)
    lg = th.logits(X[900:])
    assert (lg.argmax(1) == y[900:]).mean() > 0.8
    th.temperature = fit_temperature(lg, y[900:])
    assert 0.25 <= th.temperature <= 8.0
    assert np.allclose(th.probs(X[900:]).sum(1), 1.0)


def test_archive_streaming_and_decode():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "imgs.tar.gz")
    with tarfile.open(path, "w:gz") as tar:
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            buf = io.BytesIO()
            Image.new("RGB", (1024, 747), (10, 20, 30)).save(buf, "JPEG")
            data = buf.getvalue()
            info = tarfile.TarInfo(f"eccv_18_all_images_sm/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    got = dict(cct20.iter_archive(path, {"a.jpg", "c.jpg"}))
    assert set(got) == {"a.jpg", "c.jpg"}
    im = cct20.decode(got["a.jpg"], 160, 224)
    assert im.size == (224, 160)
    assert tuple(to_batch([im, im]).shape) == (2, 3, 160, 224)


def test_gain_table():
    spec = importlib.util.spec_from_file_location(
        "train_exits", os.path.join(ROOT, "pipeline", "02_train_exits.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(1)
    n = 2000
    conf = rng.random(n)
    correct_tri = rng.random(n) < conf
    correct_ref = rng.random(n) < 0.9
    edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0 + 1e-9])
    table, counts = mod.gain_table(conf, correct_tri, correct_ref, rng.random(n) < 0.2,
                                   np.ones(n), edges, 20.0)
    assert table.shape == (4, 2) and (table >= 0).all() and counts.sum() == n
    assert table[0, 0] > table[-1, 0], "unsure triage frames should gain more from refinement"
