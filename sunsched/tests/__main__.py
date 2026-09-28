"""Run the tests without pytest.

    python -m sunsched.tests            # fast unit tests, seconds, no data needed
    python -m sunsched.tests --slow     # also run pipeline stages 4-7 on a synthetic world
"""
import importlib
import sys
import traceback

MODULES = ["sunsched.tests.test_battery", "sunsched.tests.test_forecast",
           "sunsched.tests.test_simulator", "sunsched.tests.test_vision", "sunsched.tests.test_data"]


def main():
    failed = total = 0
    for name in MODULES:
        mod = importlib.import_module(name)
        for attr in sorted(dir(mod)):
            if attr.startswith("test_") and callable(getattr(mod, attr)):
                total += 1
                try:
                    getattr(mod, attr)()
                    print(f"  PASS  {name.split('.')[-1]}::{attr}")
                except Exception:
                    failed += 1
                    print(f"  FAIL  {name.split('.')[-1]}::{attr}")
                    traceback.print_exc()
    print(f"\n{total - failed}/{total} unit tests passed")
    if "--slow" in sys.argv:
        from sunsched.tests import smoke_pipeline
        total += 1
        if smoke_pipeline.main() != 0:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
