
"""
Revised evaluation for EMBARIA's simulation-based Risk_Level classification.

IMPORTANT:
Risk_Level is generated from a composite formula in
build_master_dataset.py. This evaluates reproduction of constructed
categories, NOT real-world infiltration prediction.

Raw input variables still contain information used to create the target.
Therefore, this is not independent-outcome validation.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

TARGET = "Risk_Level"

TARGET_ORDER = ["Low", "Medium", "High", "Critical"]
TARGET_CODES = [0, 1, 2, 3]

DEFAULT_DATASET_PATH = "data/processed/master_risk_dataset.csv"
DEFAULT_MODEL_DIR = "models/revised"


# Exclude target-derived scores, identifiers, coordinates and metadata.
# Raw inputs are retained because this is explicitly a constructed-label
# classification task, not independent real-world outcome prediction.
EXCLUDE = {
    "Zone_ID",
    "Latitude",
    "Longitude",
    "Land_Cover",
    "Risk_Score",
    "Risk_Level",
    "Environmental_Risk_Score",
    "Operational_Risk_Score",
    "Historical_Risk_Score",
}


# ============================================================
# 2. MODEL DEFINITIONS
# ============================================================

def make_models(seed=42):
    """Create the models evaluated on identical CV splits."""

    return {
        "Majority baseline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),

        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight=None,
                random_state=seed,
            )),
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),

        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=2.0,
                objective="multi:softprob",
                num_class=4,
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=1,
            )),
        ]),
    }


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def align_probabilities(model, X_validation):
    """
    Align predict_proba columns with TARGET_ORDER.

    This prevents class-probability columns from being interpreted
    in the wrong order.
    """

    raw_probabilities = model.predict_proba(X_validation)

    aligned = np.zeros(
        (len(X_validation), len(TARGET_ORDER)),
        dtype=float,
    )

    fitted_model = model.named_steps["model"]

    for column_index, class_code in enumerate(fitted_model.classes_):
        aligned[:, int(class_code)] = raw_probabilities[:, column_index]

    return aligned


def calculate_multiclass_auc(y_true, probabilities):
    """Calculate weighted one-vs-rest multiclass ROC-AUC."""

    return roc_auc_score(
        y_true,
        probabilities,
        labels=TARGET_CODES,
        multi_class="ovr",
        average="weighted",
    )


def verify_evaluation_outputs(y_true, predictions_by_model,
                              reports, confusion_matrices):
    """
    Verify that the classification reports and confusion matrices
    were produced from the same out-of-fold predictions.
    """

    expected_samples = len(y_true)

    for model_name, y_pred in predictions_by_model.items():
        matrix = np.asarray(
            confusion_matrices[model_name]["matrix"],
            dtype=int,
        )

        report = reports[model_name]

        # Check that the confusion matrix accounts for every sample.
        matrix_total = int(matrix.sum())

        assert matrix_total == expected_samples, (
            f"{model_name}: confusion matrix contains {matrix_total} "
            f"samples; expected {expected_samples}."
        )

        # Reconstruct the confusion matrix from the saved predictions.
        reconstructed = confusion_matrix(
            y_true,
            y_pred,
            labels=TARGET_CODES,
        )

        assert np.array_equal(matrix, reconstructed), (
            f"{model_name}: saved confusion matrix does not match "
            "the out-of-fold predictions."
        )

        # Check the classification report's total support.
        report_total = int(report["weighted avg"]["support"])

        assert report_total == expected_samples, (
            f"{model_name}: report support is {report_total}; "
            f"expected {expected_samples}."
        )

        # Verify accuracy independently using the confusion matrix.
        matrix_accuracy = np.trace(matrix) / matrix_total
        report_accuracy = float(report["accuracy"])

        assert np.isclose(matrix_accuracy, report_accuracy), (
            f"{model_name}: confusion-matrix accuracy "
            f"({matrix_accuracy:.4f}) differs from report accuracy "
            f"({report_accuracy:.4f})."
        )

        # Verify class-wise precision, recall and F1.
        for index, class_name in enumerate(TARGET_ORDER):
            row_sum = matrix[index, :].sum()
            column_sum = matrix[:, index].sum()
            true_positive = matrix[index, index]

            expected_recall = (
                true_positive / row_sum if row_sum else 0.0
            )
            expected_precision = (
                true_positive / column_sum if column_sum else 0.0
            )

            denominator = expected_precision + expected_recall
            expected_f1 = (
                2 * expected_precision * expected_recall / denominator
                if denominator else 0.0
            )

            class_report = report[class_name]

            assert np.isclose(
                class_report["recall"], expected_recall
            ), (
                f"{model_name}, {class_name}: recall mismatch."
            )

            assert np.isclose(
                class_report["precision"], expected_precision
            ), (
                f"{model_name}, {class_name}: precision mismatch."
            )

            assert np.isclose(
                class_report["f1-score"], expected_f1
            ), (
                f"{model_name}, {class_name}: F1-score mismatch."
            )

    print(
        "\nJSON consistency check passed: confusion matrices and "
        "classification reports match the out-of-fold predictions."
    )


# ============================================================
# 4. TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate(
    master_dataset_path=DEFAULT_DATASET_PATH,
    model_dir=DEFAULT_MODEL_DIR,
    n_splits=5,
    seed=42,
):
    os.makedirs(model_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not os.path.exists(master_dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {master_dataset_path}\n"
            "Run the master dataset creation script first."
        )

    df = pd.read_csv(master_dataset_path)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    unknown_labels = sorted(
        set(df[TARGET].astype(str)) - set(TARGET_ORDER)
    )

    if unknown_labels:
        raise ValueError(
            f"Unexpected target labels: {unknown_labels}"
        )

    # Encode target in fixed order.
    y = pd.Categorical(
        df[TARGET],
        categories=TARGET_ORDER,
        ordered=True,
    ).codes

    if (y < 0).any():
        raise ValueError("Target encoding failed.")

    # Select numeric predictors.
    feature_cols = [
        column for column in df.columns
        if column not in EXCLUDE
    ]

    if not feature_cols:
        raise ValueError("No predictor features remain after exclusions.")

    X = df[feature_cols].copy()

    non_numeric = X.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            "Non-numeric features need encoding: "
            f"{non_numeric}"
        )

    if X.isna().all(axis=0).any():
        empty_columns = X.columns[X.isna().all()].tolist()
        raise ValueError(
            f"Features containing only missing values: {empty_columns}"
        )

    counts = (
        df[TARGET]
        .value_counts()
        .reindex(TARGET_ORDER, fill_value=0)
    )

    if counts.min() < n_splits:
        raise ValueError(
            f"Smallest class has {counts.min()} rows; "
            f"reduce n_splits below {counts.min()}."
        )

    print("=" * 65)
    print("EMBARIA — REVISED MODEL EVALUATION")
    print("=" * 65)
    print(f"Dataset: {master_dataset_path}")
    print(f"Rows: {len(df)} | Features: {len(feature_cols)}")
    print("\nTarget counts:")
    print(counts.to_string())

    print(
        "\nWARNING: Risk_Level is formula-derived. These metrics evaluate "
        "reproduction of constructed labels, not real-world "
        "infiltration prediction.\n"
    )

    # --------------------------------------------------------
    # Create consistent CV splits for every model
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    # Materialize the splits once so every model uses the same folds.
    splits = list(cv.split(X, y))

    models = make_models(seed)

    summary_rows = []
    fold_rows = []
    reports = {}
    confusion_matrices = {}
    predictions_by_model = {}
    fitted_models = {}

    # --------------------------------------------------------
    # Evaluate each model
    # --------------------------------------------------------

    for model_name, estimator in models.items():

        print(f"\nEvaluating: {model_name}")

        oof_pred = np.full(len(df), -1, dtype=int)

        oof_proba = np.full(
            (len(df), len(TARGET_ORDER)),
            np.nan,
            dtype=float,
        )

        fold_auc = []

        for fold_number, (train_indices, validation_indices) in enumerate(
            splits,
            start=1,
        ):
            model = clone(estimator)

            X_train = X.iloc[train_indices]
            X_validation = X.iloc[validation_indices]

            y_train = y[train_indices]
            y_validation = y[validation_indices]

            model.fit(X_train, y_train)

            predictions = model.predict(X_validation).astype(int)

            probabilities = align_probabilities(
                model,
                X_validation,
            )

            # Save each validation prediction exactly once.
            oof_pred[validation_indices] = predictions
            oof_proba[validation_indices] = probabilities

            # AUC is available only when all classes are present
            # in this validation fold.
            if len(np.unique(y_validation)) == len(TARGET_ORDER):
                try:
                    fold_auc_value = calculate_multiclass_auc(
                        y_validation,
                        probabilities,
                    )
                except ValueError as error:
                    fold_auc_value = np.nan
                    print(
                        f"WARNING: {model_name}, fold {fold_number}: "
                        f"AUC unavailable: {error}"
                    )
            else:
                fold_auc_value = np.nan
                print(
                    f"WARNING: {model_name}, fold {fold_number}: "
                    "not all classes are present; fold AUC unavailable."
                )

            fold_auc.append(fold_auc_value)

            fold_rows.append({
                "Model": model_name,
                "Fold": fold_number,
                "Accuracy": accuracy_score(
                    y_validation, predictions
                ),
                "Weighted_Precision": precision_score(
                    y_validation,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "Weighted_Recall": recall_score(
                    y_validation,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "Weighted_F1": f1_score(
                    y_validation,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "ROC_AUC_OVR_Weighted": fold_auc_value,
            })

        # Ensure all rows received out-of-fold predictions.
        if (oof_pred < 0).any():
            raise RuntimeError(
                f"{model_name}: some rows have no OOF prediction."
            )

        if np.isnan(oof_proba).any():
            raise RuntimeError(
                f"{model_name}: some rows have no OOF probabilities."
            )

        # ----------------------------------------------------
        # Pooled out-of-fold metrics
        # ----------------------------------------------------

        auc_oof = calculate_multiclass_auc(y, oof_proba)

        row = {
            "Model": model_name,
            "OOF_Accuracy": accuracy_score(y, oof_pred),
            "OOF_Weighted_Precision": precision_score(
                y,
                oof_pred,
                average="weighted",
                zero_division=0,
            ),
            "OOF_Weighted_Recall": recall_score(
                y,
                oof_pred,
                average="weighted",
                zero_division=0,
            ),
            "OOF_Weighted_F1": f1_score(
                y,
                oof_pred,
                average="weighted",
                zero_division=0,
            ),
            "OOF_Macro_F1": f1_score(
                y,
                oof_pred,
                average="macro",
                zero_division=0,
            ),
            "OOF_ROC_AUC_OVR_Weighted": auc_oof,
            "Fold_AUC_Mean": (
                float(np.nanmean(fold_auc))
                if np.isfinite(fold_auc).any()
                else np.nan
            ),
            "Fold_AUC_SD": (
                float(np.nanstd(fold_auc, ddof=1))
                if np.sum(np.isfinite(fold_auc)) > 1
                else np.nan
            ),
            "N_AUC_Folds_Available": int(
                np.sum(np.isfinite(fold_auc))
            ),
        }

        summary_rows.append(row)

        # Both artifacts are generated from the exact same OOF predictions.
        reports[model_name] = classification_report(
            y,
            oof_pred,
            labels=TARGET_CODES,
            target_names=TARGET_ORDER,
            output_dict=True,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y,
            oof_pred,
            labels=TARGET_CODES,
        )

        confusion_matrices[model_name] = {
            "classes": TARGET_ORDER,
            "matrix": matrix.tolist(),
        }

        predictions_by_model[model_name] = oof_pred.copy()

        # Save OOF data for later reproducibility and analysis.
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
        )

        np.savez_compressed(
            os.path.join(model_dir, f"{safe_name}_oof.npz"),
            y_true=y,
            y_pred=oof_pred,
            y_proba=oof_proba,
            target_order=np.asarray(TARGET_ORDER),
        )

        fitted_models[model_name] = estimator

    # --------------------------------------------------------
    # Save evaluation outputs
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        "OOF_Weighted_F1",
        ascending=False,
    )

    summary_df.to_csv(
        os.path.join(model_dir, "model_comparison_oof.csv"),
        index=False,
    )

    pd.DataFrame(fold_rows).to_csv(
        os.path.join(model_dir, "fold_metrics.csv"),
        index=False,
    )

    reports_path = os.path.join(
        model_dir,
        "classification_reports.json",
    )

    confusion_path = os.path.join(
        model_dir,
        "confusion_matrices.json",
    )

    with open(reports_path, "w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2)

    with open(confusion_path, "w", encoding="utf-8") as file:
        json.dump(confusion_matrices, file, indent=2)

    # --------------------------------------------------------
    # Verify JSON files before continuing
    # --------------------------------------------------------

    verify_evaluation_outputs(
        y,
        predictions_by_model,
        reports,
        confusion_matrices,
    )

    print("\nSaved files:")
    print(f"- {os.path.join(model_dir, 'model_comparison_oof.csv')}")
    print(f"- {os.path.join(model_dir, 'fold_metrics.csv')}")
    print(f"- {reports_path}")
    print(f"- {confusion_path}")

    # --------------------------------------------------------
    # Fit final Random Forest on all rows for dashboard use
    # --------------------------------------------------------
    # This final model is NOT used to calculate the CV metrics above.

    final_rf = clone(models["Random Forest"])
    final_rf.fit(X, y)

    model_artifact = {
        "model": final_rf,
        "feature_names": feature_cols,
        "target_map": {
            name: index
            for index, name in enumerate(TARGET_ORDER)
        },
        "target_order": TARGET_ORDER,
        "interpretation": (
            "Constructed Risk_Level classification; "
            "not real-world outcome prediction."
        ),
    }

    model_path = os.path.join(
        model_dir,
        "random_forest_model.joblib",
    )

    joblib.dump(model_artifact, model_path)

    # --------------------------------------------------------
    # Print final comparison
    # --------------------------------------------------------

    print("\nFINAL OUT-OF-FOLD MODEL COMPARISON")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print(f"\nSaved final Random Forest artifact to: {model_path}")
    print(f"All revised outputs saved to: {model_dir}")

    return summary_df


# ============================================================
# 5. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_and_evaluate()
