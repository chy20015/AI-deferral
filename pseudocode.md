# Data preparation and model training

These steps produce the inputs described in `README.md`.

## JustRAIGS

```
for each image:
    crop to the fundus (bounding box of grey level > 15 on a downscaled copy)
    pad to a square, resize to 512 x 512 (bicubic)

groups = union-find over images
    union(image, its fellow eye)
    union(images with identical file bytes)
features = timm 'vit_large_patch14_dinov2.lvd142m', Resize(448) + CenterCrop(448), L2-normalised embedding
for each pair of images in different groups with cosine similarity >= 0.995:
    union(pair)

table = reviewers.justraigs_labels(public label table)
table.age = 'Age', set to missing outside 18-100
table.cluster = group
table.fold = StratifiedGroupKFold(n_splits=5, shuffle=True).split(table, table.y, table.cluster)
write images.csv: image_id, cluster, fold, y, y_dis, y_u, h, age
```

## ConvNeXt-T, three heads

```
for k in 0..4, seed in 0..2:
    calibration fold c = (k + 1) mod 5; training = the three other folds
    model = timm 'convnext_tiny.fb_in22k', 3 outputs: y, y_dis, y_u
    train transform = RandomResizedCrop(512, scale 0.85-1.0, ratio 0.95-1.05), horizontal flip,
                      rotation up to 15 degrees, ColorJitter(0.2, 0.2, 0.1, 0.02), ImageNet normalisation
    evaluation transform = ImageNet normalisation
    loss = sum over heads j of weight_j * mean BCEWithLogits(pos_weight_j)
           weight = (1, 0.5, 0.5); pos_weight_j = sqrt(negatives_j / positives_j) in the training folds
           images without y_dis are masked in head 2
    AdamW(lr 1e-4, weight decay 0.05), batch 64, linear warm-up over the first epoch, cosine decay, bf16 autocast
    train 8 epochs, keep the final weights
    save pred_f{k}_s{seed}.npz: cal_ids, cal_logits (fold c), test_ids, test_logits (fold k)
```

## DINOv2-L/14 heads

```
X = DINOv2-L/14 embeddings at 448 px, Resize(448) + CenterCrop(448), frozen, not L2-normalised
for k in 0..4:
    standardise X with the mean and SD of the training folds
    for head in (linear 1024 -> 3, MLP 1024 -> 512 -> GELU -> dropout 0.2 -> 3), seed in 0..2:
        same targets, head weights and pos_weight as above
        AdamW(lr 1e-3, weight decay 1e-4), batch 1,024, cosine decay, 30 epochs, keep the final weights
        save pred_f{k}_s{seed}.npz as above
```

## AI strength

```
for each stratum (fold, group contains an RG image) of patient groups:
    rank = one random permutation of the groups in the stratum
for f in (0.05, 0.10, 0.20, 0.40):
    keep a training group if rank < round(f * stratum size)
    calibration and evaluation folds are complete
    train the ConvNeXt-T as above on the kept training groups
```

## ISIC 2018

```
y = diagnosis in {MEL, BCC, AKIEC}
h = reviewers.isic_decisions(reader table)
analysis images = test images with an image file and at least one unaided read
fold = StratifiedGroupKFold(n_splits=2, shuffle=True).split(analysis images, y, lesion_id)

for seed in 0..2:
    model = ConvNeXt-T as above with 1 output, trained on all HAM10000 images
    448 px; vertical flip and rotation up to 30 degrees added; evaluation Resize(448) + CenterCrop(448)
    20 epochs, keep the final weights
p = mean over seeds of sigmoid(logit); pvar = variance over seeds
write images.csv: image_id, cluster (lesion_id), fold, y, h, age, p, pvar
```

## External glaucoma strata

```
PAPILA: glaucoma vs healthy, suspects excluded, both eyes, cluster = patient
HYGD: first image per patient by identifier, ten patients used to pilot a reading interface excluded,
      cluster = patient
FIVES: glaucoma and normal images, cluster = image
REFUGE-Zeiss: training set; REFUGE-Canon: validation and test sets; cluster = image
preprocess as JustRAIGS
p = mean RG probability of the 15 main JustRAIGS models (5 folds x 3 seeds)
pool the strata in which the AI reaches an AUC of at least 0.70
fold: clusters in random order, assigned alternately to folds 0 and 1
write images.csv: stratum, image_id, cluster, fold, y, p
```
