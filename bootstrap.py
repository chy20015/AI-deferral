import numpy as np


def poisson_weights(clusters, n_boot, rng):
    _, index = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    rng = np.random.default_rng(rng)
    for _ in range(n_boot):
        yield rng.poisson(1.0, index.max() + 1)[index].astype(float)


def percentile(values, q):
    v = np.asarray(values, float)
    v = np.sort(v[~np.isnan(v)])
    position = q * (len(v) - 1)
    lo, hi = int(np.floor(position)), int(np.ceil(position))
    a, b = v[lo], v[hi]
    if a == b:
        return float(a)
    if np.isinf(a) or np.isinf(b):
        return float(a if q < 0.5 else b)
    return float(a + (position - lo) * (b - a))


def interval(values):
    return percentile(values, 0.025), percentile(values, 0.975)
