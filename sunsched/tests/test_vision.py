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


def _train_exits_module():
    spec = importlib.util.spec_from_file_location(
        "train_exits", os.path.join(ROOT, "pipeline", "02_train_exits.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gain_table_is_monotone_and_informative():
    mod = _train_exits_module()
    rng = np.random.default_rng(1)
    n = 3000
    p_animal = rng.random(n)
    is_animal = rng.random(n) < p_animal
    values = np.where(is_animal, 1.0, 0.1)
    # Refinement helps animal frames (species gets fixed), not empties.
    correct_tri = np.where(is_animal, rng.random(n) < 0.3, rng.random(n) < 0.9)
    correct_ref = np.where(is_animal, rng.random(n) < 0.8, rng.random(n) < 0.9)
    edges = np.linspace(0, 1, 6)
    edges[-1] += 1e-9
    table, raw, counts = mod.gain_table(p_animal, correct_tri, correct_ref, values, edges, 20.0)
    assert table.shape == (5,) and (table >= 0).all() and counts.sum() == n
    assert np.all(np.diff(table) >= -1e-12), "gain must be non-decreasing in p_animal"
    assert table[-1] > table[0] + 0.1, "likely-animal frames should gain much more"


def test_isotonic_regression():
    mod = _train_exits_module()
    y = np.array([0.1, 0.5, 0.3, 0.2, 0.9])
    w = np.ones(5)
    out = mod.isotonic_increasing(y, w)
    assert np.all(np.diff(out) >= -1e-12)
    assert np.isclose(out.sum(), y.sum()), "pooling must preserve the weighted total"
    assert np.allclose(mod.isotonic_increasing(np.arange(5.0), w), np.arange(5.0))
