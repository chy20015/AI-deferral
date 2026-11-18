# Evaluation of AI deferral with assumed and recorded reviewers

Code for the deferral policies, the system at a fixed specificity, the four reviewer models (oracle, class-conditional noise, matched simulation, recorded decisions), the gain and its decomposition, the break-even sensitivity and the bootstrap described in Methods. Image preprocessing, patient grouping, fold assignment and model training are given as pseudocode in `pseudocode.md`.

Not included: the reviewer whose errors rise with image difficulty, the resampling of reviewer decisions, the two-way reader bootstrap in ISIC 2018, the comparison of the AI with the graders, the evaluation of the AI on the retained images, the external AUCs, the descriptive analyses of the deferred images and the sensitivity analyses. No data are included.

## Files

| File | Contents | Methods |
|---|---|---|
| `policies.py` | Scores of D0 to D10 fitted on the calibration fold: threshold `t0` at 95% specificity, isotonic `q`, ROC slope `lambda`, the value of equation (1), kernel regression for D8, calibrated disagreement for D9, gradient boosting for D10. Selection of the top share of images with random tie-breaking, and cross-fitting over folds. | Model development; Deferral policies; equation (1) |
| `system.py` | System sensitivity at a fixed system specificity, with the AI threshold re-set on the retained images; the AI alone; the terms `s`, `f`, `a` and `C` and the gain; the decomposition of the matched simulation's error; the break-even sensitivity, at the reviewers' or an assumed specificity, for one stratum or summed over strata. | System sensitivity and the reviewer's contribution, equation (2); Reviewer accuracy on the deferred images, equation (3); Conditions for a gain |
| `reviewers.py` | The oracle, class-conditional noise (0.90, 0.02), the matched simulation and the recorded decisions: in JustRAIGS the mean of G1 and G2 with ungradable as referral (G3 where both are missing), in ISIC 2018 the mean over readers of each reader's mean unaided decision. | Reviewer models |
| `bootstrap.py` | Poisson weights drawn per cluster and percentile intervals; an unattainable break-even sensitivity enters as infinity. | Statistical analysis |
| `evaluate.py` | Runs the analysis for JustRAIGS, ISIC 2018 or the external strata and writes the estimates with 95% intervals: gains under the four reviewer models, their comparison, equations (2) and (3), break-even sensitivities and differences in gain between policies. | Comparison of reviewer models; Reviewer accuracy on the deferred images; Conditions for a gain; Statistical analysis |
| `pseudocode.md` | Preprocessing, patient groups, folds, ConvNeXt-T and DINOv2 models, AI strength, ISIC 2018 model, external strata. | Data; Model development |

Policy codes follow the paper: D0 random, D1 distance to the operating threshold, D2 predicted ungradable vote, D3 age, D4 entropy, D5 seed variance, D6 predicted disagreement, D7 value with a perfect reviewer, D8 value with the reviewer modelled from the AI score, D9 value with the reviewer derived from the predicted disagreement, D10 learned value. `D8e` is D8 fitted on the evaluation images; it is computed whenever D9 is and serves as its comparator.

## Requirements

Python 3.12 and the packages in `requirements.txt`. Model training used PyTorch 2.11 and timm 1.0.30.

## Inputs

JustRAIGS: `images.csv` with one row per image and the columns `image_id`, `cluster` (patient group), `fold` (0 to 4), `y` (final label RG), `y_dis` (G1 and G2 differ; empty unless both gave RG or NRG), `h` (recorded decision), `age`; and a directory of prediction files `pred_f{k}_s{seed}.npz`, one per evaluation fold `k` and seed, with `cal_ids`, `cal_logits` for calibration fold `(k + 1) mod 5` and `test_ids`, `test_logits` for fold `k`. Logits have three columns: RG, disagreement, ungradable vote.

ISIC 2018: `images.csv` with `image_id`, `cluster` (lesion), `fold` (0 or 1), `y`, `h`, `age`, `p` (mean AI probability over seeds), `pvar` (variance over seeds).

External strata: `images.csv` with `stratum`, `image_id`, `cluster`, `fold` (0 or 1), `y`, `p`.

Other columns are carried along to `deferred.csv`. The names `p`, `pvar`, `pdis`, `pu` and `t0` are set by the code in JustRAIGS, and `pdis` and `pu`, if present, are used as features of D10, so the tables should not contain other columns with these names.

## Usage

```
python evaluate.py justraigs images.csv out --predictions predictions
python evaluate.py isic images.csv out
python evaluate.py external images.csv out
```

Options: `--boot` (replicates, default 2,000) and `--seed`. Random deferral, tie-breaking and the bootstrap weights are drawn from `--seed`, so another seed changes the results by these draws only.

## Outputs

`deferred.csv` (JustRAIGS and ISIC 2018): every input column, including the reference labels and the recorded decisions, the AI scores, the fold threshold `t0` and, for every policy and budget, whether the image was deferred (`D4@0.1` is D4 at a 10% budget). It holds per-image data and is subject to the terms of the source datasets.

`summary.csv`: estimate and 95% interval of each quantity at a 10% budget and a system specificity of 0.95. Sensitivities, gains and their terms are proportions; an unattainable break-even sensitivity is written as `inf`.

- `ai_alone`: sensitivity of the AI alone.
- `gain|model|policy`: system sensitivity minus that of the AI alone, for the models `oracle`, `noise`, `matched` and `recorded`; `gain|recorded|D8e` for the comparator.
- `s|policy`, `f|policy`, `a|policy`, `C|policy`: the terms of equation (2) with the recorded decisions.
- `sensitivity_part|policy`, `specificity_part|policy`: the two terms of equation (3).
- `break_even|policy` at the recorded specificity on the deferred negatives, `margin|policy` = `s` minus it, and `break_even_0.8|policy` at an assumed specificity of 0.8.
- `oracle_overestimation`: oracle gain minus recorded gain of the policy that is best under the oracle, re-selected in every replicate; of policies with equal oracle gains, the first in `POLICIES` or `ISIC` is taken.
- `matched_overestimation`: matched minus recorded gain, averaged over the uncertainty-based policies.
- `tau|model`: Kendall's tau-b between the policy gains with the recorded decisions and under the model.
- `difference|A-B`: recorded gain of policy A minus that of policy B, for D9 and D8, D9 and D8e, and D1 and each of D8 to D10, where both are present.

External strata: `break_even_f|policy` for assumed specificities from 0.5 to 1.0 and `oracle_gain|policy`, for each stratum and for the strata pooled by summing counts, each stratum keeping its own operating point.

In every replicate the system threshold and the matched rates are re-estimated; the deferral sets, the fold thresholds, the fitted policies and the AI scores are held fixed.
