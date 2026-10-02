# Methodology notes

## Data model

The associated study used 48 boreholes and 123,197 regularly sampled depth records from the Northern Ore Cluster in the Shu-Sarysu uranium province.

Primary logs:

- gamma ray;
- apparent resistivity;
- spontaneous potential;
- depth.

Engineered attributes included the depth-normalized gamma rate of change and squared gamma.

## Grouped validation

The most important validation rule is that intervals from one borehole must not be split across training and validation.

`GroupKFold(n_splits=5)` uses `well_id` as the grouping variable. This evaluates transfer to unseen wells rather than interpolation among neighbouring samples from a well already present in training.

## Class imbalance

The research did not use SMOTE or random undersampling. Performance was evaluated under the observed class proportions using PR-based metrics and macro-averaging where appropriate.

## Weak labels

For the productive-horizon task, some intervals could be assigned weak labels from gamma thresholds. Those rows were permitted in training but excluded from validation metrics to prevent the evaluation from simply reproducing the rule that generated the weak label.

The code therefore accepts a `weak_label_mask` explicitly.

## Random Forest

The manuscript reports grouped GridSearchCV and a final reference configuration with:

- 200 trees;
- no bootstrap sampling;
- unrestricted maximum depth;
- minimum leaf size 1;
- minimum split size 2.

The package exposes that configuration through `default_random_forest()`.

## Probability quality

For productive horizons, PR-AUC is emphasized because the positive class is imbalanced. Brier score and calibration diagnostics are also important because high discrimination does not guarantee reliable probabilities.

## Profile reconstruction

Cubic-spline interpolation provides a smooth along-profile representation of modelled properties and stratigraphic trends. It is complementary to the classifier, not a replacement for geological interpretation. Thin layers and abrupt contacts can be smoothed.

## Interpretation boundary

The workflow supports classification and geological correlation. It does not replace resource estimation, grade modelling, core control or expert stratigraphic interpretation.
