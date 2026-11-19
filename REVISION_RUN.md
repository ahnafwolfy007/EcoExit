# Revision experiments: how to run them

These experiments answer the reviewers of the SunSched paper. They reuse the
artifacts of the full CCT20 run you already did (stages 1-3), so **nothing is
downloaded and stages 1-3 are not repeated**. Only the simulator runs.

## 1. Get the code

```bash
git fetch origin
git checkout paper-revision
```

Activate the same virtual environment as before (no new packages are needed).

## 2. Check it

```bash
python -m sunsched.tests
```

All tests must pass (the new ones are in `sunsched/tests/test_revision.py`).
`test_vision` needs torch, which your environment already has.

## 3. Find the right output folder

Use the folder of the **CPU** full run, the one that produced the paper's
`autonomy_runs.csv` (it holds `artifacts/outcomes.npz`). Below it is written
as `OUT`; replace it with the real path, for example `./outputs`.

```bash
python pipeline/08_revision.py --out OUT --dry-run
```

This only counts jobs. You should see about 45,600 simulated deployments.

## 4. Run

```bash
python pipeline/08_revision.py --out OUT
```

Expect roughly 1-3 hours (about 1 s per deployment divided by the number of
cores). If it stops, rerun only the missing groups, for example
`--only sens wake latency sites`. The groups, in order of importance:

| Group | Deployments | What it answers |
|---|---|---|
| coverage | 135 | Is the conformal reserve calibrated? How does it compare with simpler rules? |
| grid | 9,720 | Does one combined controller (defer + ceiling) work in both regimes? |
| sens | 17,280 | Do the claims survive every assumed hardware constant? |
| wake | 12,150 | Below what wake energy does deferral stop paying? |
| latency | 2,700 | How do the deadline and queue size trade delay for value? |
| sites | 3,600 | Do the claims hold under five other climates? |

## 5. Send back

Everything is written to `OUT/results/tables/revision/` (six `*_runs.csv`
files and `manifest.json`). Nothing earlier is overwritten. Commit that folder
to this branch and push:

```bash
git add -f OUT/results/tables/revision
git commit -m "Revision experiments: results"
git push origin paper-revision
```

## Notes

- The `grid` group re-runs the paper's own policies. Their numbers should
  match the paper's `autonomy_runs.csv` exactly (the simulator is
  deterministic). That check tells us the right artifacts were used.
- No hardware is involved anywhere.
