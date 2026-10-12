"""
More defensible SHAP + permutation importance evaluation.
Run after the revised cross-validation script. SHAP is still explanatory,
not evidence of causality or real-world validity.
"""
import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import clone

TARGET_ORDER = ["Low", "Medium", "High", "Critical"]
EXCLUDE = {"Zone_ID","Latitude","Longitude","Land_Cover","Risk_Score","Risk_Level",
           "Environmental_Risk_Score","Operational_Risk_Score","Historical_Risk_Score"}

def evaluate_explainability(dataset="data/processed/master_risk_dataset.csv",
                            outdir="models/revised/explainability", seed=42):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(dataset).dropna(subset=["Risk_Level"]).reset_index(drop=True)
    X = df[[c for c in df.columns if c not in EXCLUDE]].copy()
    y = pd.Categorical(df["Risk_Level"], categories=TARGET_ORDER, ordered=True).codes

    # Cross-fold permutation importance on held-out data; lower F1 after
    # shuffling a feature means the model relied on that feature in that fold.
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    model_template = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                         class_weight="balanced", random_state=seed,
                                         n_jobs=-1))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    perm_rows = []
    fold_models = []
    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        model = clone(model_template).fit(X.iloc[tr], y[tr])
        fold_models.append(model)
        base_f1 = f1_score(y[va], model.predict(X.iloc[va]), average="weighted", zero_division=0)
        perm = permutation_importance(model, X.iloc[va], y[va], scoring="f1_weighted",
                                      n_repeats=20, random_state=seed+fold, n_jobs=-1)
        for i, feature in enumerate(X.columns):
            perm_rows.append({"Fold":fold, "Feature":feature,
                              "Permutation_Importance_Mean":perm.importances_mean[i],
                              "Permutation_Importance_SD":perm.importances_std[i],
                              "Fold_Baseline_Weighted_F1":base_f1})
    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(os.path.join(outdir, "permutation_importance_by_fold.csv"), index=False)
    perm_summary = perm_df.groupby("Feature").agg(
        Mean_Importance=("Permutation_Importance_Mean","mean"),
        SD_Across_Folds=("Permutation_Importance_Mean","std")
    ).sort_values("Mean_Importance", ascending=False)
    perm_summary.to_csv(os.path.join(outdir, "permutation_importance_summary.csv"))

    # SHAP explanations for a fold model on held-out observations only.
    # Use one fold's model and held-out rows for an illustrative explanation;
    # do not call this a complete stability analysis.
    tr, va = next(iter(cv.split(X, y)))
    model = clone(model_template).fit(X.iloc[tr], y[tr])
    tree = model.named_steps["model"]
    X_eval = X.iloc[va]
    explainer = shap.TreeExplainer(tree)
    shap_values = explainer.shap_values(model.named_steps["imputer"].transform(X_eval))

    # Normalize SHAP output shape across SHAP versions.
    if isinstance(shap_values, list):
        vals = np.stack(shap_values, axis=-1) # samples, features, classes
    else:
        vals = np.asarray(shap_values)
        if vals.ndim == 2:
            vals = vals[:, :, None]
        # Some versions can return (samples, classes, features).
        if vals.shape[1] == len(TARGET_ORDER) and vals.shape[2] == X.shape[1]:
            vals = np.transpose(vals, (0,2,1))
    global_imp = np.mean(np.abs(vals), axis=(0,2))
    shap_summary = pd.DataFrame({"Feature":X.columns,
                                 "Mean_Absolute_SHAP":global_imp}).sort_values(
                                     "Mean_Absolute_SHAP", ascending=False)
    shap_summary.to_csv(os.path.join(outdir, "shap_global_importance_heldout_fold.csv"), index=False)

    plt.figure()
    plt.barh(shap_summary.head(15)["Feature"][::-1],
             shap_summary.head(15)["Mean_Absolute_SHAP"][::-1])
    plt.xlabel("Mean absolute SHAP value (model output scale)")
    plt.title("SHAP global importance — held-out fold")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "shap_global_importance.png"), dpi=200)
    plt.close()

    joined = shap_summary.merge(perm_summary.reset_index(), on="Feature", how="outer")
    joined.to_csv(os.path.join(outdir, "shap_vs_permutation.csv"), index=False)
    print("Saved explainability outputs to:", outdir)
    print(joined.sort_values("Mean_Absolute_SHAP", ascending=False).head(15).to_string(index=False))
    print("Interpretation caution: correlated inputs can split or distort importance.")
    print("SHAP/permutation describe model behavior, not causality or real-world validity.")

if __name__ == "__main__":
    evaluate_explainability()
