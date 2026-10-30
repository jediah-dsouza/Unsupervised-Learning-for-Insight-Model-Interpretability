# Task 8 deliverables

Files:
- `clustering_pca.py`: scaling/encoding, K-Means candidate k=2..6, silhouette + inertia, cluster profiles, PCA plot and diagnostics.
- `interpretability.py`: loads an existing fitted pipeline, creates stratified holdout, compares built-in and permutation importance, attempts SHAP summary.
- `INSIGHTS.md`: observed cluster profiles and interpretation of model drivers.
- `outputs/`: generated tables, plots, assignments, and importance CSVs.
- `churnguard-xgboost-v1.joblib`: fitted pipeline extracted from the supplied ChurnGuard project archive for reproducible interpretation. If your Task 7 model differs, replace the `--model` argument with that exact artifact.

Run:
```bash
python clustering_pca.py --data /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv --out outputs
python interpretability.py --data /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv --model churnguard-xgboost-v1.joblib --out outputs
```
The code requires recent scikit-learn (for `OneHotEncoder(sparse_output=False)`), pandas, numpy, matplotlib, joblib, and SHAP.
