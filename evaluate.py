import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from bootstrap import interval, poisson_weights
from policies import POLICIES, UNCERTAINTY, crossfit
from reviewers import reviewer_models
from system import (ai_alone, ai_flagged, break_even, break_even_counts, gain_terms, oracle_gain, ranking,
                    simulation_error, system_sensitivity)

ISIC = ('D0', 'D1', 'D7', 'D4', 'D5', 'D8', 'D10', 'D3')
EXTERNAL = ('D1', 'D4')
BUDGET = 0.1
ASSUMED_SPECIFICITY = 0.8
SPECIFICITY_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)
CONTRASTS = (('D9', 'D8'), ('D9', 'D8e'), ('D1', 'D8'), ('D1', 'D9'), ('D1', 'D10'))


def load_justraigs(images, prediction_dir, folds=5):
    images = images.set_index('image_id')
    pairs = []
    for k in range(folds):
        files = sorted(f for f in os.listdir(prediction_dir) if f.startswith(f'pred_f{k}_s') and f.endswith('.npz'))
        runs = [np.load(os.path.join(prediction_dir, f), allow_pickle=True) for f in files]
        pair = []
        for part in ('cal', 'test'):
            if any((run[f'{part}_ids'] != runs[0][f'{part}_ids']).any() for run in runs):
                raise ValueError(f'image ids differ between the prediction files of fold {k}')
            prob = np.stack([1 / (1 + np.exp(-run[f'{part}_logits'])) for run in runs])
            table = images.loc[runs[0][f'{part}_ids']].reset_index()
            table['p'], table['pvar'] = prob[:, :, 0].mean(0), prob[:, :, 0].var(0)
            table['pdis'], table['pu'] = prob[:, :, 1].mean(0), prob[:, :, 2].mean(0)
            pair.append(table)
        pairs.append(tuple(pair))
    return pairs


def split_folds(table, folds=2):
    return [(table[table['fold'] != k], table[table['fold'] == k]) for k in range(folds)]


def statistics(pool, policies, w, order, budget=BUDGET, assumed=ASSUMED_SPECIFICITY):
    y, h = pool['y'].to_numpy(), pool['h'].to_numpy(float)
    models = reviewer_models(y, h, w)
    alone = ai_alone(y, w, order)
    flagged = ai_flagged(y, w, order)
    out = {'ai_alone': alone}
    for name in policies:
        d = pool[f'{name}@{budget}'].to_numpy(bool)
        for model, h_model in models.items():
            out[f'gain|{model}|{name}'] = system_sensitivity(y, h_model, d, w, order) - alone
        recorded = gain_terms(y, h, d, w, order, flagged)
        matched = gain_terms(y, models['matched'], d, w, order, flagged)
        out[f'sensitivity_part|{name}'], out[f'specificity_part|{name}'] = simulation_error(recorded, matched)
        for key in ('s', 'f', 'a', 'C'):
            out[f'{key}|{name}'] = recorded[key]
        s_star = break_even(break_even_counts(y, d, recorded['f'], w, order))
        out[f'break_even|{name}'] = s_star
        out[f'margin|{name}'] = recorded['s'] - s_star
        out[f'break_even_{assumed}|{name}'] = break_even(break_even_counts(y, d, assumed, w, order))
    if f'D8e@{budget}' in pool:
        d = pool[f'D8e@{budget}'].to_numpy(bool)
        out['gain|recorded|D8e'] = system_sensitivity(y, h, d, w, order) - alone
    for first, second in CONTRASTS:
        if f'gain|recorded|{first}' in out and f'gain|recorded|{second}' in out:
            out[f'difference|{first}-{second}'] = out[f'gain|recorded|{first}'] - out[f'gain|recorded|{second}']
    gains = {model: np.array([out[f'gain|{model}|{name}'] for name in policies]) for model in models}
    best = int(np.argmax(gains['oracle']))
    uncertain = [i for i, name in enumerate(policies) if name in UNCERTAINTY]
    out['oracle_overestimation'] = gains['oracle'][best] - gains['recorded'][best]
    out['matched_overestimation'] = float(np.mean(gains['matched'][uncertain] - gains['recorded'][uncertain]))
    for model in ('oracle', 'noise', 'matched'):
        out[f'tau|{model}'] = kendalltau(gains['recorded'], gains[model]).statistic
    return out


def analyse(pool, policies, n_boot, rng, budget=BUDGET):
    order = ranking(pool['p'].to_numpy())
    point = statistics(pool, policies, np.ones(len(pool)), order, budget)
    replicates = {key: [] for key in point}
    for w in poisson_weights(pool['cluster'], n_boot, rng):
        for key, v in statistics(pool, policies, w, order, budget).items():
            replicates[key].append(v)
    rows = [(key, point[key], *interval(replicates[key])) for key in point]
    return pd.DataFrame(rows, columns=['quantity', 'estimate', 'lower', 'upper'])


def external_counts(pool, w, budget=BUDGET):
    y = pool['y'].to_numpy()
    order = ranking(pool['p'].to_numpy())
    return {(name, f): break_even_counts(y, pool[f'{name}@{budget}'].to_numpy(bool), f, w, order)
            for name in EXTERNAL for f in SPECIFICITY_GRID}


def external_quantities(counts):
    out = {}
    for (name, f), c in counts.items():
        out[f'break_even_{f}|{name}'] = break_even(c)
        if f == 1.0:
            out[f'oracle_gain|{name}'] = oracle_gain(c)
    return out


def external_analysis(pools, n_boot, seed):
    point, replicates = {}, {}
    for i, (name, pool) in enumerate(pools.items()):
        point[name] = external_counts(pool, np.ones(len(pool)))
        replicates[name] = [external_counts(pool, w) for w in poisson_weights(pool['cluster'], n_boot, [seed, i])]
    keys = list(point[next(iter(pools))])
    point['pooled'] = {k: sum(point[name][k] for name in pools) for k in keys}
    replicates['pooled'] = [{k: sum(replicates[name][r][k] for name in pools) for k in keys} for r in range(n_boot)]
    rows = []
    for group in point:
        estimate = external_quantities(point[group])
        boot = [external_quantities(c) for c in replicates[group]]
        for key, v in estimate.items():
            rows.append((group, key, v, *interval([b[key] for b in boot])))
    return pd.DataFrame(rows, columns=['stratum', 'quantity', 'estimate', 'lower', 'upper'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['justraigs', 'isic', 'external'])
    parser.add_argument('images')
    parser.add_argument('out')
    parser.add_argument('--predictions')
    parser.add_argument('--boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.dataset == 'justraigs' and args.predictions is None:
        parser.error('justraigs needs --predictions')
    os.makedirs(args.out, exist_ok=True)
    images = pd.read_csv(args.images)
    rng = np.random.default_rng(args.seed)
    if args.dataset == 'external':
        pools = {name: crossfit(split_folds(group), rng, EXTERNAL, (BUDGET,))
                 for name, group in images.groupby('stratum', sort=False)}
        result = external_analysis(pools, args.boot, args.seed)
    else:
        if args.dataset == 'justraigs':
            pairs, policies = load_justraigs(images, args.predictions), POLICIES
        else:
            pairs, policies = split_folds(images), ISIC
        pool = crossfit(pairs, rng, policies)
        pool.to_csv(os.path.join(args.out, 'deferred.csv'), index=False)
        result = analyse(pool, policies, args.boot, rng)
    result.to_csv(os.path.join(args.out, 'summary.csv'), index=False)


if __name__ == '__main__':
    main()
