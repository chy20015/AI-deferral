import numpy as np
import pandas as pd

REFERRAL = {'RG': 1.0, 'U': 1.0, 'NRG': 0.0}
MALIGNANT = {'MEL', 'BCC', 'AKIEC'}


def oracle(y):
    return y.astype(float)


def class_conditional_noise(y, sensitivity=0.90, specificity=0.98):
    return np.where(y == 1, sensitivity, 1 - specificity)


def matched_simulation(y, h, w):
    pos, neg = y == 1, y == 0
    sensitivity = (w * pos * h).sum() / (w * pos).sum()
    false_positive_rate = (w * neg * h).sum() / (w * neg).sum()
    return np.where(pos, sensitivity, false_positive_rate)


def reviewer_models(y, h, w):
    return {'oracle': oracle(y), 'noise': class_conditional_noise(y),
            'matched': matched_simulation(y, h, w), 'recorded': h}


def justraigs_labels(labels):
    g1, g2, g3 = (labels[f'Label G{i}'].map(REFERRAL) for i in (1, 2, 3))
    both = labels['Label G1'].isin(['RG', 'NRG']) & labels['Label G2'].isin(['RG', 'NRG'])
    return pd.DataFrame({
        'image_id': labels['Eye ID'],
        'y': (labels['Final Label'] == 'RG').astype(int),
        'y_dis': np.where(both, (labels['Label G1'] != labels['Label G2']).astype(float), np.nan),
        'y_u': (labels['Label G1'].eq('U') | labels['Label G2'].eq('U')).astype(int),
        'h': pd.concat([g1, g2], axis=1).mean(axis=1).fillna(g3),
    })


def isic_decisions(reads):
    unaided = reads[reads['probType'] == 'no help']
    malignant = unaided['answer'].str.upper().isin(MALIGNANT).astype(float)
    per_reader = malignant.groupby([unaided['image_isic_id'], unaided['tequ_user_id']]).mean()
    return per_reader.groupby(level=0).mean()
