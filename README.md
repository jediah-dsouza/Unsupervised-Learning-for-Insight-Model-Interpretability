# Customer Segmentation & Model Interpretability

**Unsupervised Learning, Customer Analytics & Explainable AI**

A machine learning project that combines customer segmentation with model interpretability to uncover meaningful customer groups and understand the key factors associated with telecommunications customer churn.

The project uses **K-Means clustering, Principal Component Analysis (PCA), feature importance, permutation importance, and SHAP** to turn model outputs into interpretable analytical insights.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Output Files](#output-files)
- [Business Value](#business-value)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Overview

Customer churn is an important challenge for telecommunications companies. Understanding customer behavior, identifying distinct customer segments, and explaining model predictions can help businesses make better-informed retention decisions.

This project explores the **Telco Customer Churn dataset** through two complementary analytical approaches:

1. **Customer Segmentation:** Applies K-Means clustering to identify groups of customers with similar characteristics, using PCA to visualize their distribution in a lower-dimensional space.
2. **Model Interpretability:** Evaluates feature importance using an existing fitted classification pipeline, permutation importance, and SHAP where supported.

The objective is to combine unsupervised learning with explainable machine learning to produce interpretable findings that can support customer analytics and retention strategy development.

## Objectives

- Prepare numerical and categorical customer features for clustering.
- Evaluate candidate K-Means configurations for cluster quality.
- Compare clustering performance using inertia and silhouette scores.
- Profile customer segments and examine their observed churn rates.
- Use PCA to visualize clusters and investigate dimensionality reduction.
- Compare built-in model feature importance with permutation importance.
- Generate SHAP-based explanations when compatible with the fitted model.
- Document findings and translate analytical results into potential business applications.

## Methodology

### 1. Data Preparation

The clustering workflow prepares customer features through scaling and categorical encoding.

Preprocessing is designed to make mixed numerical and categorical data suitable for distance-based clustering and subsequent analysis.

### 2. K-Means Customer Segmentation

K-Means clustering is evaluated across candidate cluster counts from **k = 2 to k = 6**.

The analysis examines:

- **Inertia:** Measures within-cluster sum of squared distances and helps assess compactness.
- **Silhouette score:** Evaluates how well-separated and internally consistent the clusters are.
- **Cluster profiles:** Summarize customer characteristics within each cluster.
- **Observed churn rates:** Compare the proportion of customers who churn across segments.

The final choice of cluster count should consider both quantitative metrics and the practical interpretability of the resulting segments.

### 3. Principal Component Analysis (PCA)

PCA reduces the transformed feature space to two principal components for visualization.

The resulting plots help explore how customer segments are distributed and whether clusters appear distinct in the reduced-dimensional representation.

PCA is used primarily for visualization and diagnostics; separation in two dimensions does not necessarily imply strong separation in the full feature space.

### 4. Model Interpretability

The interpretability workflow loads an existing fitted classification pipeline and evaluates it using a stratified holdout split.

Three approaches are used:

| Technique | Purpose |
|---|---|
| Built-in feature importance | Examines feature importance provided by the fitted model, where supported. |
| Permutation importance | Measures how predictive performance changes when a feature is shuffled. |
| SHAP | Estimates feature contributions to model predictions when supported by the model and preprocessing pipeline. |

Comparing multiple techniques provides a broader view of model behavior than relying on a single importance measure.

**Reproducibility note:** The default model artifact is `churnguard-xgboost-v1.joblib`, extracted from the supplied ChurnGuard project archive. If your Task 7 model is different, use the corresponding fitted artifact when running the interpretability analysis.

## Key Findings

The following findings are based on the current analysis outputs.

### Customer Segmentation

- **Three clusters** were identified as the best candidate among the tested cluster counts according to the highest observed silhouette score.
- The best observed silhouette score was approximately **0.238**, indicating limited cluster separation rather than strongly distinct customer groups.
- The observed churn rates differed across clusters:

| Cluster | Observed Churn Rate |
|---|---:|
| Cluster 0 | 44.01% |
| Cluster 1 | 7.40% |
| Cluster 2 | 15.16% |

These differences suggest that the identified segments may have different retention profiles. Cluster membership alone, however, does not establish why customers churn.

### PCA

The first two principal components explained approximately:

| Component | Explained Variance |
|---|---:|
| PC1 | 27.2% |
| PC2 | 18.1% |
| **Combined** | **45.3%** |

The two-dimensional PCA visualization therefore represents approximately 45.3% of the variance in the transformed feature space.

### Model Interpretability

On the current stratified holdout evaluation, the model achieved an **average precision score of approximately 0.6617**.

The importance analyses highlighted the following factors:

- Month-to-month contract status
- Customer tenure
- Availability of online security services
- Internet service characteristics

Tenure and contract-related features were particularly prominent in the permutation importance analysis, while SHAP also highlighted month-to-month contracts, tenure, and the absence of online security.

These findings are consistent with the possibility that customer commitment and service characteristics are useful indicators of churn risk. They should be interpreted as predictive associations, not evidence of causation.

> **Important:** The reported results describe the current analysis artifacts. Results may differ if the dataset, preprocessing, fitted model, random seed, or evaluation procedure changes.

## Repository Structure

```text
Unsupervised-Learning-for-Insight-Model-Interpretability/
│
├── README.md
├── INSIGHTS.md
├── clustering_pca.py
├── interpretability.py
├── .gitignore
│
└── outputs/
    ├── cluster_profiles.csv
    ├── customer_cluster_assignments.csv
    ├── k_elbow.png
    ├── k_selection_scores.csv
    ├── k_silhouette.png
    ├── pca_clusters.png
    ├── pca_coordinates.csv
    ├── pca_top_loadings.csv
    ├── builtin_feature_importance.csv
    ├── builtin_feature_importance.png
    ├── permutation_importance.csv
    ├── permutation_importance.png
    ├── shap_mean_abs_importance.csv
    ├── shap_summary.png
    └── importance_comparison.csv
```

**Note:** The dataset and `.joblib` model artifacts are excluded from the published repository by `.gitignore`. You must obtain the dataset and appropriate fitted model artifact separately to reproduce the analysis.

## Technologies Used

- **Python** — analysis and implementation
- **pandas** — data manipulation
- **NumPy** — numerical operations
- **scikit-learn** — preprocessing, K-Means, PCA, evaluation, and permutation importance
- **Matplotlib** — data visualization
- **XGBoost** — model family used by the supplied ChurnGuard artifact, if applicable
- **SHAP** — model explanation
- **joblib** — loading fitted model artifacts
- **Git and GitHub** — version control and project hosting

## Installation

### Prerequisites

- Python with a recent version of scikit-learn
- The Telco Customer Churn CSV dataset
- A compatible fitted model artifact for the interpretability workflow

### 1. Clone the repository

```bash
git clone https://github.com/jediah-dsouza/Unsupervised-Learning-for-Insight-Model-Interpretability.git
cd Unsupervised-Learning-for-Insight-Model-Interpretability
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install pandas numpy matplotlib scikit-learn joblib shap
```

Install any additional dependencies required by your specific fitted model, such as XGBoost when applicable.

The scripts use `OneHotEncoder(sparse_output=False)`, so ensure your scikit-learn version supports this parameter.

## Reproducing the Analysis

Download or otherwise obtain the dataset and place it at a convenient local path.

### Run Customer Segmentation and PCA

```bash
python clustering_pca.py \
  --data /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --out outputs
```

On Windows PowerShell, you can also run the command on one line:

```powershell
python clustering_pca.py --data "C:\path\to\WA_Fn-UseC_-Telco-Customer-Churn.csv" --out outputs
```

This workflow generates cluster evaluation metrics, customer cluster assignments, cluster profiles, PCA diagnostics, and visualization files.

### Run Model Interpretability

Obtain the compatible fitted model artifact and place it in your working directory, or supply its full path.

```bash
python interpretability.py \
  --data /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --model churnguard-xgboost-v1.joblib \
  --out outputs
```

**Windows PowerShell:**

```powershell
python interpretability.py --data "C:\path\to\WA_Fn-UseC_-Telco-Customer-Churn.csv" --model "C:\path\to\churnguard-xgboost-v1.joblib" --out outputs
```

The model artifact must be compatible with the script's expected pipeline, input features, preprocessing, and target definition.

Both scripts accept the arguments shown above; use `--help` to inspect their available options:

```bash
python clustering_pca.py --help
python interpretability.py --help
```

## Output Files

| Output | Description |
|---|---|
| `cluster_profiles.csv` | Summary statistics for each customer segment |
| `customer_cluster_assignments.csv` | Customer-level cluster assignments |
| `k_selection_scores.csv` | Evaluation metrics for candidate cluster counts |
| `k_elbow.png` | Inertia-based cluster selection diagnostic |
| `k_silhouette.png` | Silhouette-based cluster selection diagnostic |
| `pca_clusters.png` | Two-dimensional PCA cluster visualization |
| `pca_coordinates.csv` | PCA coordinates for analyzed records |
| `pca_top_loadings.csv` | Features with prominent PCA loadings |
| `builtin_feature_importance.csv` | Built-in model importance values |
| `builtin_feature_importance.png` | Visualization of built-in feature importance |
| `permutation_importance.csv` | Permutation importance results |
| `permutation_importance.png` | Visualization of permutation importance |
| `shap_mean_abs_importance.csv` | Mean absolute SHAP importance values |
| `shap_summary.png` | SHAP summary visualization |
| `importance_comparison.csv` | Comparison of feature importance approaches |

Some interpretability outputs may be unavailable if the fitted model or its preprocessing pipeline is incompatible with the corresponding explanation method.

## Business Value

The analytical workflow demonstrates several potential applications in customer retention and service strategy.

- **Retention prioritization:** Use cluster profiles to identify segments with higher observed churn rates.
- **Customer understanding:** Examine how tenure, contract arrangements, and service characteristics differ across customer groups.
- **Model transparency:** Compare multiple importance techniques to better understand the signals used by a predictive model.
- **Decision support:** Use interpretable findings to inform further investigation into customer experience and retention opportunities.

These analyses support investigation and prioritization; they do not establish that changing a particular feature will reduce churn.

## Limitations

- The observed silhouette score indicates limited separation between clusters, so customer segments should not be treated as perfectly distinct groups.
- PCA displays only two dimensions and does not capture all information in the transformed feature space.
- Feature importance and SHAP values describe model behavior and predictive associations, not causal effects.
- Average precision depends on the holdout sample and churn prevalence; it should be considered alongside other appropriate evaluation metrics.
- Results depend on preprocessing choices, feature selection, model configuration, and random seeds.
- Reproducing the interpretability results requires the same compatible fitted model artifact or a documented replacement.

## Future Improvements

- Evaluate additional clustering algorithms, such as hierarchical clustering or DBSCAN.
- Assess cluster stability across different samples and random seeds.
- Add cluster-level visualizations and a more detailed customer segmentation report.
- Evaluate churn prediction using precision, recall, F1-score, ROC-AUC, and precision-recall curves as appropriate.
- Compare model explanations across multiple fitted models.
- Validate retention hypotheses using controlled experiments or suitable causal analysis.
- Add a dependency file to make environment setup more reproducible.

---

