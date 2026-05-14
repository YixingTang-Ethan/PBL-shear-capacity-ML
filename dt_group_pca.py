# -*- coding: utf-8 -*-
"""Decision Tree model with train-only PCA and configuration-based validation.

This script trains and evaluates a Decision Tree regressor for predicting the
shear capacity of PBL connectors. It uses configuration-based group splitting to
avoid leakage from repeated or parallel specimens, and fits all data-dependent
preprocessing steps only on the training subset.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


TARGET_COL = "Pu"
ID_COL = "NO."
REF_COL = "Ref."
FEATURE_COLS = ["nh", "Sr", "Sc", "h", "d", "ds", "fsy", "fpy", "fc", "tc"]
PCA_GROUPS = [("PC1_nh_Sr", ["nh", "Sr"]), ("PC2_h_tc", ["h", "tc"])]
PCA_INPUT_COLS = ["nh", "Sr", "h", "tc"]
PASSTHROUGH_COLS = [col for col in FEATURE_COLS if col not in PCA_INPUT_COLS]
TRANSFORMED_FEATURE_NAMES = [name for name, _ in PCA_GROUPS] + PASSTHROUGH_COLS
GROUP_COLS = FEATURE_COLS.copy()

PARAM_GRID = {
    "model__criterion": ["squared_error", "friedman_mse", "poisson"],
    "model__max_depth": [4, 5, 6, 7, 8],
    "model__max_leaf_nodes": [16, 20, 24, 28, 32, None],
    "model__max_features": [4, 6, 8, None],
    "model__min_samples_split": [2, 4, 8],
    "model__min_samples_leaf": [1, 2, 4],
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Decision Tree model with train-only PCA."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/data_after_selection.xlsx",
        help="Path to the input Excel file.",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="0",
        help="Excel sheet name or index. Default: 0.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dt_group_pca",
        help="Directory for output files.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test-set ratio.")
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits for inner CV.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=100,
        help="Number of repeated group-based train/test splits.",
    )
    parser.add_argument(
        "--group-round-decimals",
        type=int,
        default=None,
        help="Optional rounding precision when defining configuration groups.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a small parameter grid for a quick test run.",
    )
    return parser.parse_args()



def parse_sheet(sheet_value: str) -> str | int:
    try:
        return int(sheet_value)
    except ValueError:
        return sheet_value



def get_param_grid(fast: bool = False) -> dict:
    if not fast:
        return PARAM_GRID
    return {
        "model__criterion": ["poisson"],
        "model__max_depth": [6],
        "model__max_leaf_nodes": [24],
        "model__max_features": [6],
        "model__min_samples_split": [2],
        "model__min_samples_leaf": [1],
    }



def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)



def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }



def print_metrics(metrics: dict[str, float], label: str) -> None:
    print(f"{label} R2   : {metrics['R2']:.4f}")
    print(f"{label} RMSE : {metrics['RMSE']:.4f}")
    print(f"{label} MAE  : {metrics['MAE']:.4f}")
    print(f"{label} MAPE : {metrics['MAPE']:.2f} %")



def make_group_id(
    df: pd.DataFrame,
    group_cols: list[str],
    round_decimals: Optional[int] = None,
) -> pd.Series:
    group_data = df[group_cols].copy()
    if round_decimals is not None:
        numeric_cols = group_data.select_dtypes(include=[np.number]).columns
        group_data[numeric_cols] = group_data[numeric_cols].round(round_decimals)
    return group_data.astype(str).agg("|".join, axis=1)



def build_pipeline(random_seed: int) -> Pipeline:
    transformers = []
    for pc_name, cols in PCA_GROUPS:
        transformers.append(
            (
                pc_name,
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=1)),
                    ]
                ),
                cols,
            )
        )
    transformers.append(("raw_features", "passthrough", PASSTHROUGH_COLS))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", DecisionTreeRegressor(random_state=random_seed)),
        ]
    )



def get_pca_summary(fitted_pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    rows = []
    for pc_name, cols in PCA_GROUPS:
        pc_pipeline = preprocessor.named_transformers_[pc_name]
        scaler = pc_pipeline.named_steps["scaler"]
        pca = pc_pipeline.named_steps["pca"]
        for col, coef, mean, scale in zip(
            cols, pca.components_[0], scaler.mean_, scaler.scale_
        ):
            rows.append(
                {
                    "PC": pc_name,
                    "Original feature": col,
                    "PCA coefficient on standardized variable": float(coef),
                    "Training mean used for scaling": float(mean),
                    "Training std used for scaling": float(scale),
                    "Explained variance ratio": float(pca.explained_variance_ratio_[0]),
                }
            )
    return pd.DataFrame(rows)



def transformed_dataframe(fitted_pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    transformed = fitted_pipeline.named_steps["preprocess"].transform(X)
    return pd.DataFrame(transformed, columns=TRANSFORMED_FEATURE_NAMES, index=X.index)



def load_data(input_path: Path, sheet: str | int, output_dir: Path, round_decimals: Optional[int]) -> pd.DataFrame:
    df = pd.read_excel(input_path, sheet_name=sheet)
    required_cols = [ID_COL, REF_COL] + FEATURE_COLS + [TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the input file: {missing_cols}")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    df = df[df[TARGET_COL] > 0].reset_index(drop=True)
    df["group_id"] = make_group_id(df, GROUP_COLS, round_decimals)

    print("=== Data summary ===")
    print("Total samples:", len(df))
    print("Unique configuration groups:", df["group_id"].nunique())
    print("Repeated groups:", (df.groupby("group_id").size() > 1).sum())
    print("Max samples in one group:", df.groupby("group_id").size().max())

    df[[ID_COL, REF_COL] + FEATURE_COLS + [TARGET_COL, "group_id"]].to_csv(
        output_dir / "group_definition_check.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return df



def save_predictions(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
    output_dir: Path,
) -> None:
    base_cols = [ID_COL, REF_COL] + FEATURE_COLS + [TARGET_COL, "group_id"]

    train_pred = df.iloc[train_idx][base_cols].copy()
    train_pred["Predicted"] = y_train_pred
    train_pred["Residual"] = train_pred[TARGET_COL] - train_pred["Predicted"]
    train_pred["Set"] = "Train"

    test_pred = df.iloc[test_idx][base_cols].copy()
    test_pred["Predicted"] = y_test_pred
    test_pred["Residual"] = test_pred[TARGET_COL] - test_pred["Predicted"]
    test_pred["Set"] = "Test"

    pd.concat([train_pred, test_pred], axis=0).sort_index().to_csv(
        output_dir / "dt_main_group_split_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )



def save_boxplot(values: pd.Series, ylabel: str, output_stem: Path) -> None:
    plt.figure(figsize=(6.0, 4.2), dpi=300)
    plt.boxplot([values.dropna()], labels=["Decision Tree"])
    plt.ylabel(ylabel)
    plt.title("Repeated group-based validation")
    plt.tight_layout()
    plt.savefig(output_stem.with_suffix(".tiff"), dpi=600)
    plt.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close()



def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(
        input_path=input_path,
        sheet=parse_sheet(args.sheet),
        output_dir=output_dir,
        round_decimals=args.group_round_decimals,
    )

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    groups = df["group_id"].copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Data leakage detected: {len(overlap)} groups appear in both train and test.")

    print("\n=== Main group split ===")
    print("Train samples:", len(train_idx), "Test samples:", len(test_idx))
    print("Train groups :", len(train_groups), "Test groups :", len(test_groups))
    print("Group overlap:", len(overlap))

    split_record = df[[ID_COL, REF_COL] + FEATURE_COLS + [TARGET_COL, "group_id"]].copy()
    split_record["Set"] = "Train"
    split_record.loc[test_idx, "Set"] = "Test"
    split_record.to_csv(output_dir / "main_group_split_record.csv", index=False, encoding="utf-8-sig")

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    groups_train = groups.iloc[train_idx].copy()

    inner_splits = min(args.cv_splits, groups_train.nunique())
    if inner_splits < 2:
        raise ValueError("Too few training groups for GroupKFold CV.")

    grid_search = GridSearchCV(
        estimator=build_pipeline(args.seed),
        param_grid=get_param_grid(args.fast),
        cv=GroupKFold(n_splits=inner_splits),
        scoring={
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        },
        refit="r2",
        verbose=2,
        return_train_score=True,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train, groups=groups_train)

    best_pipeline = grid_search.best_estimator_
    best_idx = grid_search.best_index_

    print("\n=== Group-aware CV summary for best Decision Tree pipeline ===")
    print("Best parameters:", grid_search.best_params_)
    print("CV Train R2   :", grid_search.cv_results_["mean_train_r2"][best_idx])
    print("CV Train RMSE :", -grid_search.cv_results_["mean_train_rmse"][best_idx])
    print("CV Train MAE  :", -grid_search.cv_results_["mean_train_mae"][best_idx])
    print("CV Test R2    :", grid_search.cv_results_["mean_test_r2"][best_idx])
    print("CV Test RMSE  :", -grid_search.cv_results_["mean_test_rmse"][best_idx])
    print("CV Test MAE   :", -grid_search.cv_results_["mean_test_mae"][best_idx])

    pd.DataFrame(grid_search.cv_results_).to_csv(
        output_dir / "dt_groupaware_gridsearch_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(output_dir / "dt_best_params.json", "w", encoding="utf-8") as f:
        json.dump(grid_search.best_params_, f, indent=2)

    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)
    train_metrics = regression_metrics(y_train, y_train_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)

    print("\n=== Main group-split Decision Tree result ===")
    print_metrics(train_metrics, "Train")
    print_metrics(test_metrics, "Test")

    pd.DataFrame([
        {"Set": "Train", **train_metrics},
        {"Set": "Test", **test_metrics},
    ]).to_csv(output_dir / "dt_main_group_split_metrics.csv", index=False, encoding="utf-8-sig")

    save_predictions(df, train_idx, test_idx, y_train_pred, y_test_pred, output_dir)

    transformed_dataframe(best_pipeline, X_train).to_csv(
        output_dir / "X_train_transformed_for_DT_check.csv",
        index=False,
        encoding="utf-8-sig",
    )
    transformed_dataframe(best_pipeline, X_test).to_csv(
        output_dir / "X_test_transformed_for_DT_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pca_summary = get_pca_summary(best_pipeline)
    pca_summary.to_csv(output_dir / "train_only_pca_coefficients.csv", index=False, encoding="utf-8-sig")
    print("\n=== Train-only PCA summary ===")
    print(pca_summary)

    with open(output_dir / "dt_group_pca_pipeline.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)

    best_params_for_model = {
        key.replace("model__", ""): value
        for key, value in grid_search.best_params_.items()
        if key.startswith("model__")
    }

    print("\n=== Repeated configuration-based group validation ===")
    repeat_records = []
    repeated_splitter = GroupShuffleSplit(
        n_splits=args.n_repeats,
        test_size=args.test_size,
        random_state=args.seed,
    )

    for split_id, (tr_idx, te_idx) in enumerate(repeated_splitter.split(X, y, groups=groups), start=1):
        pipe = build_pipeline(random_seed=args.seed + split_id)
        pipe.set_params(**{f"model__{key}": value for key, value in best_params_for_model.items()})
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])

        metrics = regression_metrics(y.iloc[te_idx], pipe.predict(X.iloc[te_idx]))
        repeat_records.append(
            {
                "Split": split_id,
                "n_train": len(tr_idx),
                "n_test": len(te_idx),
                "n_train_groups": groups.iloc[tr_idx].nunique(),
                "n_test_groups": groups.iloc[te_idx].nunique(),
                **metrics,
            }
        )

        if split_id == 1 or split_id % 10 == 0:
            print(
                f"Split {split_id:03d}: R2={metrics['R2']:.4f}, "
                f"RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, "
                f"MAPE={metrics['MAPE']:.2f}%"
            )

    repeat_df = pd.DataFrame(repeat_records)
    repeat_df.to_csv(output_dir / "dt_repeated_group_validation_metrics.csv", index=False, encoding="utf-8-sig")

    print("\nRepeated validation summary:")
    print(repeat_df[["R2", "RMSE", "MAE", "MAPE"]].describe())

    save_boxplot(repeat_df["R2"], "Test R²", output_dir / "repeated_group_validation_R2_boxplot")
    save_boxplot(repeat_df["RMSE"], "Test RMSE", output_dir / "repeated_group_validation_RMSE_boxplot")

    print("\nSaved outputs in:", output_dir)


if __name__ == "__main__":
    main()
