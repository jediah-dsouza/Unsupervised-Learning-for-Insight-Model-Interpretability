#!/usr/bin/env python3
"""Interpret an already-fitted Task 7 scikit-learn pipeline on held-out data."""
from pathlib import Path
import argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="WA_Fn-UseC_-Telco-Customer-Churn.csv")
    p.add_argument("--model",default="churnguard-xgboost-v1.joblib",help="Path to the fitted Task 7 model/pipeline")
    p.add_argument("--out",default="outputs")
    p.add_argument("--target",default="Churn")
    args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.data); df.columns=df.columns.str.strip()
    y=df[args.target].map({"No":0,"Yes":1})
    X=df.drop(columns=[args.target,"customerID"],errors="ignore").copy()
    if "TotalCharges" in X: X["TotalCharges"]=pd.to_numeric(X["TotalCharges"],errors="coerce")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    model=joblib.load(args.model)
    # Do not fit or retrain: only score the existing model. Pipeline must accept raw feature columns.
    pred=model.predict_proba(X_test)[:,1]
    print(f"Holdout average precision: {average_precision_score(y_test,pred):.4f}")
    # Find the fitted estimator in a Pipeline or use model directly.
    estimator=model.steps[-1][1] if hasattr(model,"steps") else model
    preprocessor=model[:-1] if hasattr(model,"steps") else None
    # Built-in tree importance (when exposed by final estimator).
    builtin=None
    if hasattr(estimator,"feature_importances_"):
        vals=estimator.feature_importances_
        names=None
        if preprocessor is not None and hasattr(preprocessor,"get_feature_names_out"):
            try: names=preprocessor.get_feature_names_out()
            except Exception: pass
        if names is None or len(names)!=len(vals): names=[f"feature_{i}" for i in range(len(vals))]
        builtin=pd.DataFrame({"feature":names,"builtin_importance":vals}).sort_values("builtin_importance",ascending=False)
        builtin.to_csv(out/"builtin_feature_importance.csv",index=False)
        plt.figure(figsize=(9,7)); top=builtin.head(20).sort_values("builtin_importance")
        plt.barh(top.feature.astype(str),top.builtin_importance); plt.xlabel("Tree impurity/gain importance")
        plt.title("Built-in feature importance (top 20)"); plt.tight_layout(); plt.savefig(out/"builtin_feature_importance.png",dpi=160); plt.close()
    # Permutation on original held-out features; scoring reflects generalization.
    perm=permutation_importance(model,X_test,y_test,n_repeats=10,random_state=42,scoring="average_precision",n_jobs=-1)
    pi=pd.DataFrame({"feature":X_test.columns,"permutation_mean":perm.importances_mean,"permutation_std":perm.importances_std}).sort_values("permutation_mean",ascending=False)
    pi.to_csv(out/"permutation_importance.csv",index=False)
    plt.figure(figsize=(9,7)); top=pi.head(20).sort_values("permutation_mean")
    plt.barh(top.feature,top.permutation_mean,xerr=top.permutation_std); plt.xlabel("Decrease in average precision after shuffling")
    plt.title("Permutation importance on held-out data (top 20)"); plt.tight_layout(); plt.savefig(out/"permutation_importance.png",dpi=160); plt.close()
    if builtin is not None:
        # Aggregate encoded built-in importances back to raw column prefixes where possible.
        agg=[]
        for col in X.columns:
            matching=builtin[builtin.feature.astype(str).str.startswith(col+"_") | (builtin.feature.astype(str)==col)]
            agg.append({"feature":col,"builtin_aggregated":matching.builtin_importance.sum()})
        ag=pd.DataFrame(agg)
        comp=ag.merge(pi,on="feature",how="outer").fillna(0).sort_values("permutation_mean",ascending=False)
        comp.to_csv(out/"importance_comparison.csv",index=False)
    # SHAP: explain fitted estimator in transformed feature space when model is a pipeline.
    try:
        import shap
        if preprocessor is not None:
            Xt=preprocessor.transform(X_test)
            try: names=preprocessor.get_feature_names_out()
            except Exception: names=[f"feature_{i}" for i in range(Xt.shape[1])]
        else:
            Xt=X_test
            names=X_test.columns.tolist()
        # Limit rows to keep runtime and plot legible; no fitting occurs.
        n=min(500,len(X_test))
        Xt_sample=Xt[:n]
        explainer=shap.TreeExplainer(estimator)
        sv=explainer(Xt_sample)
        vals=sv.values
        # Binary classifiers can return (n, features, classes); select positive class.
        if vals.ndim==3: vals=vals[:,:,1]
        plt.figure()
        shap.summary_plot(vals,Xt_sample,feature_names=names,show=False,max_display=20)
        plt.tight_layout(); plt.savefig(out/"shap_summary.png",dpi=180,bbox_inches="tight"); plt.close()
        imp=pd.DataFrame({"feature":names,"mean_abs_shap":np.abs(vals).mean(axis=0)}).sort_values("mean_abs_shap",ascending=False)
        imp.to_csv(out/"shap_mean_abs_importance.csv",index=False)
        print("\nTop SHAP drivers:\n",imp.head(10).to_string(index=False))
    except Exception as e:
        warnings.warn(f"SHAP explanation failed for this model/dependency combination: {e}")
        (out/"shap_error.txt").write_text(str(e),encoding="utf-8")
    print("\nBuilt-in top features:\n",builtin.head(10).to_string(index=False) if builtin is not None else "Estimator does not expose feature_importances_.")
    print("\nPermutation top features:\n",pi.head(10).to_string(index=False))
    print(f"\nOutputs saved to {out.resolve()}")
if __name__=="__main__": main()
