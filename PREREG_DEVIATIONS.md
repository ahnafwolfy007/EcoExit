# Deviations from PREREGISTRATION.md

PREREGISTRATION.md is not edited. Every departure from it is listed here, with
when it was decided and what had been seen at that point.

## 1. Simulator version (decided 19 Nov 2026, before any Serengeti download)

The pre-registration was committed next to the SunSched v2 code (commit
7b7b6ba), which simulates each deployment twice and scores the second pass.
The paper uses the earlier simulator (branch `paper-revision`, from 9954545),
which scores a single pass from 80% initial charge. The held-out test uses the
paper's simulator so that its results are comparable with the paper's. H2 and
H3 are decided with the registered cells, margins and tests; H4 and H5
concern SunSched v2 and are not tested.

## 2. Evaluation-image budget (decided 19 Nov 2026, after reading metadata only)

Season S10 has 685,481 images at 167 locations. With the registered
`serengeti_eval_budget` of 15,000 images, the registered selection rule admits
only 4 official `val` cameras, below the registered minimum of 6, and stops.
The budget is raised in steps of 5,000 to the smallest value that admits 6
cameras; the ranking (most animal images first, ties by name) is unchanged.
At this point only the season metadata had been read: no image had been
downloaded and no result computed. The value used is printed in the stage 0
log as `[deviation] serengeti_eval_budget raised ...`.
