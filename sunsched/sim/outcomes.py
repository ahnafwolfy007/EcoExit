"""Per-frame classifier outcomes and refinement-gain tables, as the simulator uses them."""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class Outcomes:
    class_names: List[str]
    labels: np.ndarray                  # true class index per evaluation frame
    values: np.ndarray                  # value of a correct label, per frame
    locations: np.ndarray
    pred: Dict[str, np.ndarray]         # op -> predicted class per frame ("triage", "lite", "full")
    conf: Dict[str, np.ndarray]
    macs: Dict[str, int]
    gain_edges: np.ndarray
    gain_tables: Dict[str, np.ndarray]  # refine op -> (n_bins, 2): [bin, triage said empty]
    empty_idx: int

    def gain(self, op: str, pred: int, conf: float) -> float:
        """Expected value gained by refining a frame, given only its triage output.

        Estimated on the calibration split (a camera the simulated deployments
        never use), so no simulated frame's label informs its own gain.
        """
        b = int(np.searchsorted(self.gain_edges, conf, side="right")) - 1
        b = min(max(b, 0), self.gain_tables[op].shape[0] - 1)
        return float(self.gain_tables[op][b, int(pred == self.empty_idx)])


def load_outcomes(path: str) -> Outcomes:
    d = np.load(path, allow_pickle=False)
    names = [str(x) for x in d["class_names"]]
    ops = [str(x) for x in d["ops"]]
    refine_ops = [o for o in ops if o != "triage"]
    return Outcomes(
        class_names=names,
        labels=d["eval_labels"].astype(np.int64),
        values=d["eval_values"].astype(np.float64),
        locations=d["eval_locations"].astype(str),
        pred={o: d[f"pred_{o}"].astype(np.int64) for o in ops},
        conf={o: d[f"conf_{o}"].astype(np.float64) for o in ops},
        macs={o: int(d[f"macs_{o}"]) for o in ops},
        gain_edges=d["gain_edges"].astype(np.float64),
        gain_tables={o: d[f"gain_{o}"].astype(np.float64) for o in refine_ops},
        empty_idx=names.index("empty"),
    )
