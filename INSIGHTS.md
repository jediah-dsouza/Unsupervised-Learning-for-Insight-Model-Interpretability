# Task 8 — Customer Segmentation & Model Interpretability

## 1. Reproducibility and Scope

The analysis was performed using the provided Telco Customer Churn dataset, which contains 7,043 rows. The `Churn` target and `customerID` identifier were excluded from the clustering and PCA input features. The churn target was used only after clustering to calculate and interpret the churn rates of the resulting segments.

The supplied, fitted ChurnGuard XGBoost pipeline was reused for model interpretability without retraining. A stratified 80/20 split with `random_state=42` was used for held-out permutation importance and model evaluation.

The model achieved a holdout average precision score of **0.6617** in this run. This is a development-stage result rather than an untouched final performance estimate.

## 2. Selecting the Number of Clusters

K-Means clustering was applied to a feature matrix containing median-imputed and standardized numeric features alongside one-hot-encoded categorical features. Candidate cluster counts from k=2 to k=6 were evaluated using inertia and silhouette scores.

| Number of clusters (k) | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 68,651.1 | 0.234 |
| 3 | 55,588.5 | 0.238 |
| 4 | 50,815.3 | 0.230 |
| 5 | 47,872.9 | 0.195 |
| 6 | 45,698.3 | 0.185 |

The selected solution is **k=3**, which achieved the highest silhouette score (approximately 0.238) among the tested candidates.

The score is relatively modest, indicating that the clusters overlap rather than forming sharply separated groups. The clusters should therefore be treated as useful summaries of customer behavior, not as naturally distinct customer types.

Inertia decreases as the number of clusters increases, so it should not be used alone to select k. The silhouette score provides an additional measure of how well-separated and internally consistent the clusters are.

## 3. Cluster Profiles and Business Interpretation

### Cluster 0 — Shorter-Tenure, Higher-Risk Customers

- **Customers:** 3,188 (45.3% of the dataset)
- **Churn rate:** 44.01%
- **Average tenure:** 15.3 months
- **Average monthly charges:** 67.85
- **Common characteristics:** Month-to-month contracts, fiber-optic internet, and electronic-check payments.

This is the highest-churn segment and the clearest priority for retention analysis. Its relatively short average tenure and high churn rate suggest that early customer engagement and satisfaction may be useful areas to investigate.

Potential actions include testing targeted retention outreach and evaluating flexible contract options. Any intervention should be assessed for effectiveness and profitability rather than assuming that cluster membership itself causes churn.

### Cluster 1 — Lower-Cost Customers with Low Churn

- **Customers:** 1,526 (21.7% of the dataset)
- **Churn rate:** 7.40%
- **Average tenure:** 30.5 months
- **Average monthly charges:** 21.08
- **Common characteristics:** Two-year contracts and no internet service.

This segment has the lowest observed churn rate and the lowest average monthly charges. Its customer profile is consistent with a lower-cost segment with relatively stable retention.

Broad discount campaigns may not be necessary for this group. Instead, the business could focus on maintaining service quality and evaluating whether targeted offers provide sufficient additional value.

### Cluster 2 — Long-Tenure, Higher-Spending Customers

- **Customers:** 2,329 (33.1% of the dataset)
- **Churn rate:** 15.16%
- **Average tenure:** 56.9 months
- **Average monthly charges:** 89.15
- **Common characteristics:** Two-year contracts, fiber-optic internet, and services such as online security, online backup, device protection, technical support, and streaming.

These customers have the longest average tenure and the highest average monthly charges. Their churn rate is lower than Cluster 0's but higher than Cluster 1's.

Because this segment combines long tenure with relatively high spending, maintaining satisfaction and service quality may be important. High monthly charges alone should not be treated as evidence that a customer needs intervention.

### Overall Business Implications

Cluster 0 is the strongest candidate for targeted retention experiments because its churn rate is substantially higher than those of the other clusters. Cluster 1 has low observed churn and lower spending, while Cluster 2 represents longer-tenure, higher-spending customers whose experience may warrant continued attention.

These recommendations are hypotheses to test. Cluster associations do not establish that any particular intervention will reduce churn.

## 4. PCA Interpretation

Principal Component Analysis (PCA) was applied to the same encoded and scaled feature matrix used for clustering.

- **PC1 explains:** 27.2% of the variance.
- **PC2 explains:** 18.1% of the variance.
- **Combined variance explained:** 45.3%.

The PCA visualization is available in `outputs/pca_clusters.png`. The elbow and silhouette diagnostics are available in `outputs/k_elbow.png` and `outputs/k_silhouette.png`. The strongest component loadings are provided in `outputs/pca_top_loadings.csv`.

PCA components are weighted combinations of transformed input features rather than individual business variables. Their meaning must be inferred from the component loadings and may not have a simple business interpretation.

The two-dimensional visualization captures 45.3% of the total variance and therefore omits more than half of the variation in the data. Apparent overlap or separation in the plot may not fully represent distances in the original feature space.

## 5. Model Interpretation: Built-In Importance vs. Permutation Importance

The existing fitted XGBoost pipeline was reused without retraining.

Built-in feature importance is reported for transformed, one-hot-encoded features in `outputs/builtin_feature_importance.csv`. Held-out permutation importance is reported for the original input features in `outputs/permutation_importance.csv`. A comparison is available in `outputs/importance_comparison.csv`.

### Agreement Between Methods

Contract type, tenure, internet service, online security, and technical support appear among the important features in one or more of the interpretation methods.

### Differences in Rankings

Built-in importance ranks the month-to-month contract indicator first, with an importance of approximately 0.383.

Permutation importance ranks `tenure` first, with a mean average-precision decrease of approximately 0.113, followed by `Contract` at approximately 0.092.

The built-in importance results also rank the one-hot-encoded indicators for missing online security and fiber-optic internet highly.

### Why the Rankings Differ

Tree-based built-in importance measures how features contribute to model splits. It can distribute importance across multiple encoded categories and may favor features used frequently in splitting decisions.

Permutation importance instead measures how much model performance declines when a feature is shuffled on held-out data. When predictors are correlated, other features may retain similar information, reducing the measured impact of shuffling one feature.

The two methods also operate at different levels: built-in importance reports transformed columns, whereas permutation importance reports original input columns. Aggregating one-hot-encoded columns to their parent features is important when comparing them fairly.

Permutation importance uses average precision in this analysis, which evaluates the ranking of the positive churn class. Small or near-zero values mean that shuffling a feature did not reliably reduce this metric on the selected holdout; they do not prove that the feature is irrelevant.

## 6. SHAP Summary and Top Three Drivers

The SHAP summary plot is available in `outputs/shap_summary.png`, and mean absolute SHAP values are provided in `outputs/shap_mean_abs_importance.csv`.

The three highest-ranked features by mean absolute SHAP value were:

1. **Month-to-month contract** — 0.637774
2. **Tenure** — 0.408081
3. **No online security** — 0.255756

Mean absolute SHAP values indicate the average magnitude of a feature's contribution to model predictions. They do not, on their own, show whether a feature increases or decreases predicted churn risk. The direction and distribution of contributions should be examined in the SHAP summary plot.

In this model, month-to-month contracts tend to be associated with higher predicted churn risk, while longer tenure tends to be associated with lower predicted risk. Customers without online security also tend to receive higher-risk contributions.

These are patterns learned by the model from the available dataset. They do not establish causal relationships or prove that changing a contract or adding a service will prevent churn.

## 7. Methodological Limitations

1. **Distance-based clustering:** K-Means uses Euclidean distance. One-hot encoding many categorical features can influence distances and distort the resulting segmentation. Alternatives include Gower distance with a compatible clustering method or carefully selected and weighted features.

2. **Modest silhouette score:** The selected solution has a silhouette score of approximately 0.238, suggesting overlapping clusters and uncertain boundaries.

3. **PCA limitations:** PCA is a dimensionality-reduction and visualization technique, not a churn-prediction model. The first two components explain only 45.3% of the variance.

4. **Target exclusion:** The `Churn` target was excluded from clustering inputs and used only afterward to profile the resulting clusters. The `customerID` identifier was also excluded.

5. **Evaluation limitations:** The dataset does not provide a temporal holdout in this analysis. The stratified 80/20 split is a conventional evaluation approach, but the reported score should be treated as development-stage evidence, particularly if historical holdout data influenced prior model selection.

6. **SHAP interpretation:** SHAP explains how the fitted model uses features; it does not establish causation. Explanation stability and subgroup differences should be evaluated before using the results to drive customer-facing decisions.

## 8. Run Instructions

Install the required dependencies, then run the following commands from the project folder in PowerShell:

```powershell
python clustering_pca.py --data "WA_Fn-UseC_-Telco-Customer-Churn.csv" --out outputs

python interpretability.py --data "WA_Fn-UseC_-Telco-Customer-Churn.csv" --model "churnguard-xgboost-v1.joblib" --out outputs
```

The interpretability script reuses the supplied fitted XGBoost model without retraining it. If Task 7 used a different fitted model, replace the model path with the exact Task 7 artifact.

## 9. Conclusion

The analysis identified three customer segments with different churn rates, tenure levels, and spending patterns. Cluster 0 has the highest observed churn rate and is the clearest candidate for targeted retention experiments. Cluster 1 has the lowest churn rate and lowest average monthly charges, while Cluster 2 consists of longer-tenure, higher-spending customers.

PCA provides a two-dimensional view of the encoded feature space, although the first two components explain less than half of the total variance. Built-in feature importance, permutation importance, and SHAP provide complementary views of the fitted XGBoost model, with contract type and tenure emerging as particularly important signals.

Taken together, these findings support using segmentation to prioritize further analysis and retention experiments. The results should be validated before operational deployment, and neither cluster membership nor model explanations should be interpreted as proof of causation.