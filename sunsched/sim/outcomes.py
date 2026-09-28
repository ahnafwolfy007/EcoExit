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
    conf: Dict[str, np.ndarray]         # op -> max softmax probability
    p_animal: np.ndarray                # triage detector's probability that the frame is not empty
    macs: Dict[str, int]
    gain_edges: np.ndarray              # bin edges over p_animal
    gain_tables: Dict[str, np.ndarray]  # refine op -> expected value gain per p_animal bin
    empty_idx: int

    def gain(self, op: str, p_animal: float) -> float:
        """Expected value gained by refining a frame, given only its triage output.

        Indexed by the triage *detector's* animal probability. The first CCT20
        run indexed it by the species head's top-1 confidence instead, which
        turned out to be almost constant (0.089-0.124 for nearly every frame
        across 16 classes), so the table was noise. Estimated on calibration
        cameras that no simulated deployment uses, and made non-decreasing in
        p_animal.
        """
        t = self.gain_tables[op]
        b = int(np.searchsorted(self.gain_edges, p_animal, side="right")) - 1
        return float(t[min(max(b, 0), len(t) - 1)])


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
        p_animal=d["p_animal_triage"].astype(np.float64),
        macs={o: int(d[f"macs_{o}"]) for o in ops},
        gain_edges=d["gain_edges"].astype(np.float64),
        gain_tables={o: d[f"gain_{o}"].astype(np.float64).ravel() for o in refine_ops},
        empty_idx=names.index("empty"),
    )
