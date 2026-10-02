# Uranium Horizon ML Geophysics

**Leakage-aware Random Forest classification of productive uranium horizons from borehole geophysical logs, with geostatistical profile reconstruction.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

This repository is a clean reference implementation based on the research workflow:

> **Sharapatov, A.; Saduov, A.; Assirbek, N.A.; Abdyrov, M.**  
> *Integrating Machine Learning and Geostatistics for High-Resolution Uranium Horizon Mapping: A Case Study from the Northern Ore Cluster (Kazakhstan).*

The repository focuses on the computational methodology. It does not redistribute confidential or project-specific borehole data.

## Research question

Can conventional well logs be used to classify lithology and productive uranium-bearing intervals while avoiding the common validation error of mixing depth samples from the same borehole between training and test sets?

The key design choice is **grouped cross-validation by well**. Every interval from a held-out borehole remains outside the training fold.

## Dataset described in the manuscript

The research dataset contained:

- **48 wells** from the Northern Ore Cluster, Shu-Sarysu uranium province;
- **123,197 depth records**;
- regular vertical sampling of approximately **0.1 m**;
- gamma ray (μR/h);
- apparent resistivity (Ohm m);
- spontaneous potential (mV);
- core-derived lithology and productivity information;
- engineered gamma features.

The detailed lithology inventory was consolidated into geologically meaningful classes before classification.

## Modelling workflow

```text
Borehole logs
   |
   +--> gamma ray
   +--> apparent resistivity
   +--> spontaneous potential
   +--> depth
   |
   v
Per-well feature engineering
   +--> gamma rate of change (Δγ/Δz)
   +--> gamma²
   |
   v
Targets
   +--> lithology
   +--> productive horizon
   |
   v
Grouped 5-fold CV by well_id
   |
   +--> no SMOTE / no random interval split
   +--> weak-labelled intervals allowed in training
   +--> weak-labelled intervals excluded from validation
   |
   v
Random Forest + grouped hyperparameter tuning
   |
   v
PR-AUC + Precision/Recall/F1 + Brier/calibration
   |
   v
Along-profile cubic-spline reconstruction
   |
   v
Comparison with geological/manual interpretation
```

## Reported manuscript results

### Lithology classification

- Accuracy: **0.789**
- Macro Precision: **0.760**
- Macro Recall: **0.762**
- Macro F1: **0.761**
- Macro PR-AUC: approximately **0.79-0.80**
- Brier score: **0.139**

### Productive-horizon classification

- Accuracy: **0.969**
- Positive-class Precision: **0.866**
- Positive-class Recall: **0.900**
- Positive-class F1: **0.883**
- Average Precision: **0.94**
- Brier score: **0.067**

These are results reported by the research manuscript. The synthetic example included here is only a software demonstration.

## Leakage control

Random splitting of individual 0.1 m intervals would place neighbouring samples from the same well into both training and validation sets. Because adjacent log measurements are strongly correlated, that design can substantially overstate generalization.

This implementation therefore uses `GroupKFold` with `well_id` as the grouping key.

The research workflow also distinguished original labels from weak labels derived from gamma thresholds. Weak-labelled intervals could contribute to training but were excluded from held-out validation metrics.

## Repository structure

```text
.
├── data/README.md
├── docs/methodology.md
├── examples/synthetic_demo.py
├── src/uranium_horizon/
│   ├── __init__.py
│   ├── features.py
│   ├── interpolation.py
│   └── modeling.py
├── CITATION.cff
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/Kazinage/uranium-horizon-ml-geophysics.git
cd uranium-horizon-ml-geophysics
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python examples/synthetic_demo.py
```

## Scope and limitations

The workflow is designed for interval classification and profile-scale geological interpretation. It is not a resource-estimation system. Thin or gradational mineralized intervals may be generalized by the classifier, and cubic splines can smooth abrupt contacts. Model output must therefore be interpreted together with geological context, log quality and independent control data.

## Author

**Alisher Saduov, PhD**  
Geophysics | Uranium Exploration | GeoAI | Well-log Machine Learning  
Satbayev University, Kazakhstan  
ORCID: 0000-0003-1501-7772

## License

Code is MIT licensed. Borehole/project data are not included and remain subject to their original ownership and access conditions.
