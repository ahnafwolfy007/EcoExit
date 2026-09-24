"""Integration checks against the artifacts the pipeline writes.

These skip cleanly when the pipeline has not been run yet, so they are safe to
include in a fresh checkout. Run them after scripts/02 to confirm the leakage
fixes actually took effect in the files the simulator reads.

Run:  python -m pytest tests -q
      python tests/test_pipeline.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


class Skip(Exception):
    pass


def _need(path):
    full = os.path.join(ART, path)
    if not os.path.exists(full):
        raise Skip(f"{path} not built yet -- run scripts/01 and 02 first")
    return full


def test_evaluation_pool_is_disjoint_from_calibration():
    """Leak (c): the simulator must not score on the calibration pool."""
    d = np.load(_need("per_image_outcomes.npz"))
    if "eval_pool" not in d:
        raise Skip("artifacts predate the pool split -- re-run scripts/02")
    overlap = np.intersect1d(d["eval_pool"], d["calib_pool"])
    assert len(overlap) == 0, f"{len(overlap)} images in both pools"
    assert len(d["eval_pool"]) > 0 and len(d["calib_pool"]) > 0


def test_per_image_arrays_match_the_eval_pool():
    d = np.load(_need("per_image_outcomes.npz"))
    if "eval_pool" not in d:
        raise Skip("artifacts predate the pool split")
    n_pool = len(d["eval_pool"])
    assert d["correct"].shape[2] == n_pool, (
        f"per-image outcomes hold {d['correct'].shape[2]} images but the "
        f"evaluation pool has {n_pool} -- the simulator would index the wrong set")
    assert d["conf"].shape == d["correct"].shape


def test_controller_grid_is_validation_not_test():
    """Leak (b): the controller's value function must come from validation."""
    g = np.load(_need("final_accuracy_grid.npz"))
    if "acc_grid_val" not in g:
        raise Skip("artifacts predate the val/test grid split")
    val, test = g["acc_grid_val"], g["acc_grid"]
    assert val.shape == test.shape
    # They are computed on different data, so they must not be identical.
    assert not np.allclose(val, test), \
        "validation and test grids are identical -- the split did not take effect"


def test_confidences_are_temperature_scaled():
    """Leak (a), and the report's calibration claim.

    The report says calibration is what makes the confidence stopping rule
    meaningful. If the stored confidences are raw softmax, that claim is false.
    """
    c = np.load(_need("confidence_calibration.npz"))
    if "temps" not in c or np.ndim(c["temps"]) != 2:
        raise Skip("artifacts predate per-(resolution, exit) temperature fitting")
    temps = c["temps"]
    assert np.all(temps > 0), "non-positive temperature"
    assert not np.allclose(temps, 1.0), \
        "every temperature is 1.0 -- scaling is a no-op, so drop the claim or fix the fit"


def test_accuracy_grid_is_monotone_in_exit_depth():
    """A non-monotone grid indicates an under-converged model, not a finding.

    This is a warning rather than a hard failure: it is legitimate during a
    --quick smoke run, and only disqualifying for a reported result.
    """
    g = np.load(_need("final_accuracy_grid.npz"))
    grid = g["acc_grid_val"] if "acc_grid_val" in g else g["acc_grid"]
    bad = [i for i in range(grid.shape[0]) if np.any(np.diff(grid[i]) < -1e-3)]
    if bad:
        print(f"    WARNING: non-monotone in exit depth at resolution rows {bad}; "
              f"train longer before reporting this grid")


def test_gate_verdict_is_self_consistent():
    path = os.path.join(os.path.dirname(ART), "results", "tables", "gate_verdict.json")
    if not os.path.exists(path):
        raise Skip("gate not run yet -- run scripts/06_phase0_gate.py")
    import json
    with open(path, encoding="utf-8") as f:
        v = json.load(f)
    assert v["passed"] == (v["lambda_live"] and v["regime_binds"]
                           and v["beats_static_max"])
    if v["passed"]:
        assert v["lambda_mean"] > 0
        assert v["lambda_frac_zero"] < 0.5


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = skipped = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Skip as e:
            skipped += 1
            print(f"  SKIP  {fn.__name__}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed - skipped}/{len(fns)} passed, {skipped} skipped")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
