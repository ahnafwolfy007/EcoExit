"""Value-life hypervolume (PLAN_V2 section 5). Reference point (0, 0)."""
import numpy as np

LIFE_CAP = 20.0


def front(points):
    """Non-dominated (x, y) points, x ascending."""
    best, out = -np.inf, []
    for x, y in sorted(map(tuple, points), key=lambda p: (-p[0], -p[1])):
        if y > best:
            out.append((x, y))
            best = y
    return out[::-1]


def hypervolume(points) -> float:
    """Area dominated by the points inside the first quadrant."""
    area, prev = 0.0, 0.0
    for x, y in front(points):
        area += max(x - prev, 0.0) * max(y, 0.0)
        prev = max(x, prev)
    return float(area)


def normalise(value, value_full, life):
    return np.asarray(value) / np.maximum(value_full, 1e-9), np.minimum(life, LIFE_CAP) / LIFE_CAP
