# AI deferral with assumed and recorded reviewers

Analysis code for the paper. Data preparation and model training are given as pseudocode in `pseudocode.md`. No data are included.

```
pip install -r requirements.txt
python evaluate.py justraigs images.csv out --predictions predictions
python evaluate.py isic images.csv out
python evaluate.py external images.csv out
```

`images.csv` has one row per image:

- JustRAIGS: `image_id`, `cluster`, `fold`, `y`, `y_dis`, `h`, `age`; `predictions` holds `pred_f{k}_s{seed}.npz` with `cal_ids`, `cal_logits`, `test_ids`, `test_logits`.
- ISIC 2018: `image_id`, `cluster`, `fold`, `y`, `h`, `age`, `p`, `pvar`.
- External: `stratum`, `image_id`, `cluster`, `fold`, `y`, `p`.

Estimates with 95% intervals are written to `out/summary.csv`.
