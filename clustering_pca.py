#!/usr/bin/env python3
"""K-Means segmentation and PCA visualization for Telco Customer Churn."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data", default="WA_Fn-UseC_-Telco-Customer-Churn.csv")
    p.add_argument("--out", default="outputs")
    p.add_argument("--max-k", type=int, default=6)
    args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.data)
    df.columns=df.columns.str.strip()
    if "customerID" in df: ids=df["customerID"].astype(str)
    else: ids=pd.Series([f"row_{i}" for i in range(len(df))])
    if "Churn" not in df: raise ValueError("Expected target column Churn.")
    y=df["Churn"].astype(str).str.strip()
    X=df.drop(columns=[c for c in ["Churn","customerID"] if c in df]).copy()
    if "TotalCharges" in X: X["TotalCharges"]=pd.to_numeric(X["TotalCharges"], errors="coerce")
    numeric=X.select_dtypes(include=np.number).columns.tolist()
    categorical=[c for c in X.columns if c not in numeric]
    num_pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    cat_pipe=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep=ColumnTransformer([("num",num_pipe,numeric),("cat",cat_pipe,categorical)], verbose_feature_names_out=False)
    Z=prep.fit_transform(X)
    feature_names=prep.get_feature_names_out()
    # Compare multiple k values using both inertia and silhouette.
    rows=[]
    for k in range(2, max(3,args.max_k)+1):
        km=KMeans(n_clusters=k, random_state=42, n_init=20)
        labels=km.fit_predict(Z)
        rows.append({"k":k,"inertia":float(km.inertia_),"silhouette":float(silhouette_score(Z,labels, sample_size=min(2500,len(Z)), random_state=42))})
    scores=pd.DataFrame(rows); scores.to_csv(out/"k_selection_scores.csv",index=False)
    # Choose maximum silhouette; use elbow plot as supporting diagnostic.
    chosen=int(scores.loc[scores.silhouette.idxmax(),"k"])
    model=KMeans(n_clusters=chosen,random_state=42,n_init=20)
    labels=model.fit_predict(Z)
    df_out=pd.DataFrame({"customerID":ids,"cluster":labels,"Churn":y})
    df_out.to_csv(out/"customer_cluster_assignments.csv",index=False)
    # Business profiles: numeric means and churn rate, category modes.
    profile=[]
    for cl in sorted(np.unique(labels)):
        mask=labels==cl
        part=X.loc[mask]
        row={"cluster":int(cl),"customers":int(mask.sum()),"share_pct":round(mask.mean()*100,2),
             "churn_rate_pct":round((y.loc[mask].str.lower()=="yes").mean()*100,2)}
        for c in numeric:
            row[f"mean_{c}"]=round(float(pd.to_numeric(part[c],errors="coerce").mean()),3)
        for c in categorical:
            modes=part[c].mode(dropna=True)
            row[f"common_{c}"]=str(modes.iloc[0]) if len(modes) else "Unknown"
        profile.append(row)
    prof=pd.DataFrame(profile); prof.to_csv(out/"cluster_profiles.csv",index=False)
    # PCA is fitted on the same standardized/encoded feature representation.
    pca=PCA(n_components=2,random_state=42)
    coords=pca.fit_transform(Z)
    pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"cluster":labels,"Churn":y}).to_csv(out/"pca_coordinates.csv",index=False)
    plt.figure(figsize=(9,6))
    for cl in sorted(np.unique(labels)):
        m=labels==cl
        plt.scatter(coords[m,0],coords[m,1],s=12,alpha=.55,label=f"Cluster {cl}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("Telco customer clusters projected onto first two PCs")
    plt.legend(); plt.tight_layout(); plt.savefig(out/"pca_clusters.png",dpi=160); plt.close()
    plt.figure(figsize=(7,4))
    plt.plot(scores.k,scores.inertia,marker="o")
    plt.xlabel("Number of clusters (k)"); plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.title("K-Means elbow diagnostic"); plt.tight_layout(); plt.savefig(out/"k_elbow.png",dpi=160); plt.close()
    plt.figure(figsize=(7,4))
    plt.plot(scores.k,scores.silhouette,marker="o")
    plt.xlabel("Number of clusters (k)"); plt.ylabel("Silhouette score")
    plt.title("K-Means silhouette diagnostic"); plt.tight_layout(); plt.savefig(out/"k_silhouette.png",dpi=160); plt.close()
    loadings=pd.DataFrame(pca.components_.T,index=feature_names,columns=["PC1_loading","PC2_loading"])
    loadings["max_abs_loading"]=loadings[["PC1_loading","PC2_loading"]].abs().max(axis=1)
    loadings.sort_values("max_abs_loading",ascending=False).to_csv(out/"pca_top_loadings.csv")
    print(f"Chosen k={chosen} by highest silhouette score")
    print(scores.to_string(index=False))
    print("\nCluster profiles:\n",prof.to_string(index=False))
    print(f"\nPCA explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
if __name__=="__main__": main()
