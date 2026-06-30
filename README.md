<p align="center">
  <img src="assets/banner.png" alt="Project Banner" width="100%">
</p>

# 🧪 Composition-Driven Prediction of Thermoelectric Figure of Merit (ZT)
### Benchmarked Machine Learning, Stacked Ensemble Learning & Similarity-Guided Inverse Design

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-success)
![Material Informatics](https://img.shields.io/badge/Material-Informatics-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

</p>

---

## 📖 Overview

This repository contains our **B.Tech Semester Project** on applying **Machine Learning for Thermoelectric Materials Discovery**.

The project proposes a complete **composition-driven material informatics pipeline** that predicts the **thermoelectric figure of merit (ZT)** directly from material composition using multiple machine learning models, stacked ensemble learning, explainable AI (SHAP), and a similarity-guided inverse design framework for generating promising thermoelectric candidates.

Rather than relying solely on expensive experimental screening, this work demonstrates how modern machine learning techniques can accelerate the exploration of high-performance thermoelectric materials.

---

## 🚀 Project Highlights

✅ Composition-based feature engineering using **Pymatgen**

✅ Benchmarking of multiple Machine Learning and Deep Learning models

✅ Stacked Ensemble Learning for improved prediction accuracy

✅ SHAP Explainability for model interpretation

✅ Similarity-Guided Inverse Design

✅ Physics-aware candidate filtering

✅ Residual analysis & Cross Validation

---

# 📂 Workflow

```text
Experimental Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering (Pymatgen)
        │
        ▼
Exploratory Data Analysis
(PCA • Correlation Heatmap)
        │
        ▼
Model Benchmarking
        │
        ▼
Stacked Ensemble Learning
        │
        ▼
SHAP Explainability
        │
        ▼
Similarity Guided
Inverse Design
        │
        ▼
Candidate Thermoelectric Materials
```

---

# 📊 Models Implemented

The following machine learning models were trained and evaluated:

- Linear Regression
- Random Forest
- Support Vector Machine
- CatBoost Regressor
- Extra Trees Regressor
- Gradient Boosting Machine
- LightGBM
- XGBoost
- ZTNet (Residual MLP)

## Ensemble Models

- ExtraTrees + LightGBM
- XGBoost + GBM

---

# 🏆 Best Model

| Model | Test R² |
|---------|----------|
| ExtraTrees + LightGBM Stacked Ensemble | **0.9202** |

The stacked ensemble achieved the highest predictive performance among all evaluated models while maintaining strong generalization through 5-fold cross-validation.

---

# 🔬 Key Features

- Composition-driven prediction
- Automated feature engineering
- Benchmarking of multiple ML algorithms
- Ensemble learning
- SHAP feature importance
- Residual analysis
- Similarity-guided inverse design
- Physics-aware filtering
- Candidate material generation

---

# 📈 Visualizations

This repository includes:

- Correlation Heatmap
- PCA Projection
- PCA Explained Variance
- Actual vs Predicted plots
- Residual plots
- Feature Importance
- SHAP Summary plots
- SHAP Dependence plots
- Cross Validation analysis
- Inverse Design visualizations
- Candidate comparison graphs

---

# 🧠 Technologies Used

- Python
- Scikit-Learn
- LightGBM
- XGBoost
- CatBoost
- PyTorch
- SHAP
- NumPy
- Pandas
- Matplotlib
- Pymatgen

---

# 🎯 Research Contributions

This project proposes an integrated workflow that combines

- Material Informatics
- Machine Learning
- Ensemble Learning
- Explainable Artificial Intelligence
- Similarity-Guided Inverse Design

into a single framework for thermoelectric material screening.

---

# 📚 Dataset

The experimental thermoelectric dataset was curated from:

**StarryData2.0**

Feature engineering was performed using **Pymatgen**, generating physically meaningful compositional descriptors for machine learning.

---

# 📄 Project Report

The complete methodology, experiments, results, and discussion are available in the project report included in this repository.

---

# 👨‍💻 Authors

### Sudarshan

Machine Learning • Material Informatics • Feature Engineering • Model Development

### Sai Jeevan

Data Processing • Model Evaluation • Documentation

### Shatrujit

Research • Validation • Analysis

---

# 🤝 Collaborators

<table>
<tr>
<td align="center">

**Sudarshan**

Project Lead

</td>

<td align="center">

**Sai Jeevan**

Collaborator

</td>

<td align="center">

**Shatrujit**

Collaborator

</td>

</tr>
</table>

---

# 🌟 Acknowledgements

We sincerely thank our faculty members and department for their guidance and encouragement throughout this semester project.

We also acknowledge the developers and researchers behind the following open-source libraries and datasets:

- StarryData2.0
- Pymatgen
- Scikit-Learn
- LightGBM
- XGBoost
- CatBoost
- PyTorch
- SHAP

---


It motivates us to continue exploring Machine Learning for Materials Discovery.

---
