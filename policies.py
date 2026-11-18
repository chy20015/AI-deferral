import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

EPS = 1e-6
SPECIFICITY = 0.95
BUDGETS = (0.05, 0.1, 0.2)
POLICIES = ('D0', 'D1', 'D7', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D2', 'D3')
UNCERTAINTY = ('D4', 'D5', 'D6')


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def entropy(p):
    return -(p * np.log(p + EPS) + (1 - p) * np.log(1 - p + EPS))


def isotonic(x, y):
    return IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(x, y)


def kernel_regression(x, y, x_new, grid=200):
    bandwidth = 1.06 * np.std(x) * len(x) ** (-0.2) + EPS
    points = np.linspace(min(x.min(), x_new.min()), max(x.max(), x_new.max()), grid)
    fitted = np.empty(grid)
    for i, point in enumerate(points):
        k = np.exp(-0.5 * ((x - point) / bandwidth) ** 2)
        fitted[i] = (k * y).sum() / max(k.sum(), EPS)
    return np.interp(x_new, points, fitted)


def roc_slope(p, y, t0, k=200):
    negatives = np.sort(p[y == 0])
    i = np.searchsorted(negatives, t0)
    lo, hi = negatives[max(i - k, 0)], negatives[min(i + k, len(negatives) - 1)]
    tp = ((p >= lo) & (y == 1)).sum() - ((p >= hi) & (y == 1)).sum()
    fp = ((p >= lo) & (y == 0)).sum() - ((p >= hi) & (y == 0)).sum()
    return max(tp, 1) / max(fp, 1)


def value(q, a, h1, h0, lam):
    return q * (h1 - a) - lam * (1 - q) * (h0 - a)


def reviewer_by_score(p, y, h, p_new):
    pos, neg = y == 1, y == 0
    return (kernel_regression(logit(p[pos]), h[pos], logit(p_new)),
            kernel_regression(logit(p[neg]), h[neg], logit(p_new)))


def features(table):
    columns = [logit(table['p'].to_numpy())]
    for name in ('pdis', 'pu'):
        if name in table:
            columns.append(logit(table[name].to_numpy()))
    columns.append(table['pvar'].to_numpy())
    return np.column_stack(columns)


def fit_scores(cal, ev, rng, policies=POLICIES, specificity=SPECIFICITY):
    p_cal, y_cal = cal['p'].to_numpy(), cal['y'].to_numpy()
    p = ev['p'].to_numpy()
    t0 = np.quantile(p_cal[y_cal == 0], specificity)
    lam = roc_slope(p_cal, y_cal, t0)
    q = isotonic(p_cal, y_cal).predict(p)
    a = (p >= t0).astype(float)
    scores = {}
    if 'D0' in policies:
        scores['D0'] = rng.random(len(p))
    if 'D1' in policies:
        scores['D1'] = -np.abs(logit(p) - logit(np.array(t0)))
    if 'D7' in policies:
        scores['D7'] = value(q, a, 1.0, 0.0, lam)
    if 'D4' in policies:
        scores['D4'] = entropy(p)
    if 'D5' in policies:
        scores['D5'] = ev['pvar'].to_numpy()
    if 'D6' in policies:
        scores['D6'] = ev['pdis'].to_numpy()
    if 'D8' in policies:
        h1, h0 = reviewer_by_score(p_cal, y_cal, cal['h'].to_numpy(), p)
        scores['D8'] = value(q, a, h1, h0, lam)
    if 'D9' in policies:
        known = cal['y_dis'].notna().to_numpy()
        delta = isotonic(cal['pdis'].to_numpy()[known], cal['y_dis'].to_numpy()[known]).predict(ev['pdis'].to_numpy())
        scores['D9'] = value(q, a, 1 - delta / 2, delta / 2, lam)
    if 'D10' in policies:
        h_cal, a_cal = cal['h'].to_numpy(), (p_cal >= t0).astype(float)
        realised = np.where(y_cal == 1, h_cal - a_cal, -lam * (h_cal - a_cal))
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)
        scores['D10'] = model.fit(features(cal), realised).predict(features(ev))
    if 'D2' in policies:
        scores['D2'] = ev['pu'].to_numpy()
    if 'D3' in policies:
        age = ev['age'].to_numpy(float)
        scores['D3'] = np.where(np.isnan(age), np.nanmedian(age), age)
    if 'D9' in policies:
        y_ev = ev['y'].to_numpy()
        h1, h0 = reviewer_by_score(p, y_ev, ev['h'].to_numpy(), p)
        scores['D8e'] = value(isotonic(p, y_ev).predict(p), a, h1, h0, lam)
    return scores, t0


def select(score, budget, rng):
    n = len(score)
    order = np.lexsort((rng.random(n), -score))
    deferred = np.zeros(n, bool)
    deferred[order[:int(round(budget * n))]] = True
    return deferred


def crossfit(pairs, rng, policies=POLICIES, budgets=BUDGETS, specificity=SPECIFICITY):
    parts = []
    for cal, ev in pairs:
        scores, t0 = fit_scores(cal, ev, rng, policies, specificity)
        part = ev.reset_index(drop=True).copy()
        part['t0'] = t0
        deferred = {f'{name}@{b}': select(score, b, rng) for name, score in scores.items() for b in budgets}
        parts.append(pd.concat([part, pd.DataFrame(deferred)], axis=1))
    return pd.concat(parts, ignore_index=True)
