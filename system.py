import numpy as np

SPECIFICITY = 0.95
TOLERANCE = 1e-9


def ranking(p):
    return np.argsort(-p, kind='stable')


def ai_true_positives(y, kept, fp_used, w, order, specificity=SPECIFICITY):
    neg, pos = y == 0, y == 1
    allowance = (1 - specificity) * (w * neg).sum() - fp_used
    if allowance < 0:
        return 0.0
    fp = np.cumsum((w * neg * kept)[order])
    tp = np.cumsum((w * pos * kept)[order])
    n = np.searchsorted(fp, allowance + TOLERANCE, side='right')
    return tp[n - 1] if n > 0 else 0.0


def system_sensitivity(y, h, d, w, order, specificity=SPECIFICITY):
    pos, neg = y == 1, y == 0
    tp = (w * pos * d * h).sum()
    fp = (w * neg * d * h).sum()
    return (tp + ai_true_positives(y, ~d, fp, w, order, specificity)) / (w * pos).sum()


def ai_alone(y, w, order, specificity=SPECIFICITY):
    return system_sensitivity(y, y.astype(float), np.zeros(len(y), bool), w, order, specificity)


def ai_flagged(y, w, order, specificity=SPECIFICITY):
    fp = np.cumsum((w * (y == 0))[order])
    n = np.searchsorted(fp, (1 - specificity) * (w * (y == 0)).sum() + TOLERANCE, side='right')
    flagged = np.zeros(len(y), bool)
    flagged[order[:n]] = True
    return flagged


def freed_allowance(y, d, f, w, order, flagged, specificity=SPECIFICITY):
    fp_used = (1 - f) * (w * (y == 0) * d).sum()
    retained = ai_true_positives(y, ~d, fp_used, w, order, specificity)
    retained_alone = (w * (y == 1) * ~d * flagged).sum()
    return (retained - retained_alone) / (w * (y == 1)).sum()


def gain_terms(y, h, d, w, order, flagged, specificity=SPECIFICITY):
    pos, neg = y == 1, y == 0
    pos_deferred, neg_deferred = (w * pos * d).sum(), (w * neg * d).sum()
    share = pos_deferred / (w * pos).sum()
    s = (w * pos * d * h).sum() / pos_deferred
    f = 1 - (w * neg * d * h).sum() / neg_deferred
    a = (w * pos * d * flagged).sum() / pos_deferred
    c = freed_allowance(y, d, f, w, order, flagged, specificity)
    return {'s': s, 'f': f, 'a': a, 'C': c, 'share': share, 'gain': (s - a) * share + c}


def simulation_error(recorded, matched):
    sensitivity_part = (matched['s'] - recorded['s']) * recorded['share']
    specificity_part = matched['C'] - recorded['C']
    return sensitivity_part, specificity_part


def break_even_counts(y, d, f, w, order, specificity=SPECIFICITY):
    fp_used = (1 - f) * (w * (y == 0) * d).sum()
    alone = ai_true_positives(y, np.ones(len(y), bool), 0.0, w, order, specificity)
    retained = ai_true_positives(y, ~d, fp_used, w, order, specificity)
    return np.array([alone - retained, (w * (y == 1) * d).sum(), (w * (y == 1)).sum()])


def break_even(counts):
    shortfall, pos_deferred = counts[0], counts[1]
    if pos_deferred > 0:
        s = shortfall / pos_deferred
    else:
        s = 0.0 if shortfall <= TOLERANCE else np.inf
    s = max(s, 0.0)
    return np.inf if s > 1 + TOLERANCE else s


def oracle_gain(counts):
    return (counts[1] - counts[0]) / counts[2]
