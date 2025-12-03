# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import shap
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from itertools import combinations
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from statsmodels.stats.outliers_influence import variance_inflation_factor

FEATURE_NAME_MAP = {
    "phylogeny_dist_sym": "Phylogenetic Distance",
    "syntax_dist_sym": "Syntactic Distance",
    "geo_dist_sym": "Geographic Distance",
    "log_data_prop_sum": "Data Proportion (Sum)",
    "log_data_prop_min": "Data Proportion (Min)",
    "optimized": "Tailored Optimization",
    "eflomal_sym": "Eflomal Score"
}

# -----------------------------
# Data preprocessing and merging
# -----------------------------
def preprocess_and_merge(language_path, pair_path, accuracy_df, model_name, method_name, mode_name):
    language_df = pd.read_csv(language_path)
    pair_df = pd.read_csv(pair_path)

    acc_df = accuracy_df.copy()
    acc_df = acc_df.melt(id_vars=["source", "target"], var_name="setting_model", value_name="accuracy")
    acc_df[["setup", "knowledge_type", "model"]] = acc_df["setting_model"].str.extract(r"(fine-tuning|in-context_learning)_(generated|real)_new_knowledge_(.*)_accuracy")
    
    mode_tag = mode_name.replace("_new_knowledge", "")
    acc_df = acc_df[(acc_df["setup"] == method_name) & (acc_df["knowledge_type"] == mode_tag) & (acc_df["model"] == model_name)]
    
    merged = acc_df.merge(pair_df, on=["source", "target"], how="left")
    lang_src = language_df.add_suffix("_src")
    lang_tgt = language_df.add_suffix("_tgt")
    merged = merged.merge(lang_src, left_on="source", right_on="language_src", how="left")
    merged = merged.merge(lang_tgt, left_on="target", right_on="language_tgt", how="left")

    merged["log_data_prop_src"] = np.log(merged["data_proportion_src"])
    merged["log_data_prop_tgt"] = np.log(merged["data_proportion_tgt"])
    merged["log_data_prop_sum"] = merged["log_data_prop_src"] + merged["log_data_prop_tgt"]
    merged["log_data_prop_min"] = np.minimum(merged["log_data_prop_src"], merged["log_data_prop_tgt"])

    if model_name != "GPT-4o-Mini-2024-07-18":
        merged["optimized"] = merged["optimized_src"] + merged["optimized_tgt"]
    
    return merged

# -----------------------------
# Analysis (Spearman correlation)
# -----------------------------
def cor_analysis(merged_df, model_name, method_name, mode_name, output_path="results/transfer/"):
    feature_cols = [
        "phylogeny_dist_sym", "syntax_dist_sym", "phonology_dist_sym", "inventory_dist_sym", "geo_dist_sym",
        "log_data_prop_sum", "log_data_prop_min", "optimized",
        "token_overlap_jsd_sym", "eflomal_sym", "neuron_overlap_sym"
    ]
    
    print(f"\n=== Spearman correlation results for {model_name} | {method_name} | {mode_name} ===\n")
    
    results = []

    for feat in feature_cols:
        if feat not in merged_df.columns:
            continue

        feature_df = merged_df[["source", "target", "accuracy", feat]].dropna()
        feature_df = feature_df[feature_df["source"] != feature_df["target"]]

        if "asym" not in feat:
            feature_df["src_tgt_sorted"] = feature_df.apply(lambda row: tuple(sorted([row["source"], row["target"]])), axis=1)
            grouped = (feature_df.groupby("src_tgt_sorted").agg({feat: "first","accuracy": "mean"}).reset_index().rename(columns={feat: "feat_value", "accuracy": "acc_value"}))
        else:
            grouped = feature_df.rename(columns={feat: "feat_value", "accuracy": "acc_value"})
        
        r, p = spearmanr(grouped["acc_value"], grouped["feat_value"])
        print(f"[{feat}] n = {grouped.shape[0]:3d} samples  r = {r:+.3f}, p = {p:.4f}")
        results.append({
            "model": model_name,
            "method": method_name,
            "mode": mode_name,
            "feature": feat,
            "n_samples": grouped.shape[0],
            "spearman_r": r,
            "p_value": p
        })
    
    output_dir = os.path.join(output_path)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"spearman_summary_{model_name}_{method_name}_{mode_name}.csv"
    output_file = os.path.join(output_dir, filename)

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\n[✓] Saved Spearman summary to: {output_file}")

# -----------------------------
# Exhaustive Feature Selection
# -----------------------------
def adjusted_r2_score(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def exhaustive_feature_selection(merged_df, model_name, method_name, mode_name, output_path="results/transfer/"):
    # Step 1: Define all candidate features
    if model_name != 'GPT-4o-Mini-2024-07-18':
        all_features = [
            "phylogeny_dist_sym", "syntax_dist_sym", "geo_dist_sym",
            "log_data_prop_sum", "log_data_prop_min", "optimized", "eflomal_sym"
        ]
    else:
        all_features = [
            "phylogeny_dist_sym", "syntax_dist_sym", "geo_dist_sym",
            "log_data_prop_sum", "log_data_prop_min", "eflomal_sym"
        ]
    
    # Step 2: Filter out source==target and drop rows with NaNs
    df_filtered = merged_df[merged_df["source"] != merged_df["target"]].copy()
    drop_cols = all_features + ["accuracy"]
    df_filtered = df_filtered.dropna(subset=drop_cols).copy()

    # Step 3: Symmetrize language pairs
    df_filtered["lang_pair"] = df_filtered.apply(lambda row: tuple(sorted([row["source"], row["target"]])), axis=1)
    agg_dict = {feat: "mean" for feat in all_features}
    agg_dict["accuracy"] = "mean"
    df_grouped = df_filtered.groupby("lang_pair").agg(agg_dict).reset_index()

    # Step 4: Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df_grouped[all_features]), columns=all_features)
    y = df_grouped["accuracy"].values

    # Step 5: Calculate VIF and filter features
    vif_df = pd.DataFrame()
    vif_df["feature"] = all_features
    vif_df["VIF"] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]

    print("\n=== Variance Inflation Factor (VIF) ===")
    print(vif_df)

    selected_features = vif_df[vif_df["VIF"] < 10]["feature"].tolist()
    print(f"\n[✓] Selected {len(selected_features)} features with VIF < 10: {selected_features}")

    # Step 6: Exhaustive subset search using Adjusted R²
    best_adj_r2 = -np.inf
    best_subset = []
    results = []

    print("\n=== Exhaustive Subset Search (Adjusted R²) ===")
    for r in range(1, len(selected_features) + 1):
        for subset in combinations(selected_features, r):
            X_sub = X_scaled[list(subset)].values
            model = LinearRegression()
            y_pred = cross_val_predict(model, X_sub, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
            adj_r2 = adjusted_r2_score(y, y_pred, p=len(subset))
            r2 = r2_score(y, y_pred)

            print(f"Features: {subset} -> Adjusted R² = {adj_r2:.4f} (R² = {r2:.4f})")

            results.append({
                "model": model_name,
                "method": method_name,
                "mode": mode_name,
                "features": subset,
                "n_features": len(subset),
                "r2": r2,
                "adj_r2": adj_r2
            })

            if adj_r2 > best_adj_r2:
                best_adj_r2 = adj_r2
                best_subset = subset

    # Step 7: Save result
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, f"exhaustive_selection_{model_name}_{method_name}_{mode_name}.csv")
    pd.DataFrame(results).sort_values("adj_r2", ascending=False).to_csv(out_file, index=False)

    print(f"\n[✓] Best subset: {best_subset} -> Adjusted R² = {best_adj_r2:.4f}")
    print(f"[✓] Results saved to: {out_file}")

# -----------------------------
# Single-step Regression Ablation
# -----------------------------
def single_step_regression(merged_df, model_name, method_name, mode_name, output_path="results/transfer/"):
    # Step 1: Define candidate features
    if model_name != "GPT-4o-Mini-2024-07-18":
        all_features = [
            "phylogeny_dist_sym", "syntax_dist_sym", "geo_dist_sym",
            "log_data_prop_sum", "log_data_prop_min",
            "optimized", "eflomal_sym"
        ]
    else:
        all_features = [
            "phylogeny_dist_sym", "syntax_dist_sym", "geo_dist_sym",
            "log_data_prop_sum", "log_data_prop_min", "eflomal_sym"
        ]

    # Step 2: Filter data (remove source==target, drop NaNs)
    df_filtered = merged_df[merged_df["source"] != merged_df["target"]].copy()
    drop_cols = all_features + ["accuracy"]
    df_filtered = df_filtered.dropna(subset=drop_cols).copy()

    # Step 3: Symmetrize language pairs
    df_filtered["lang_pair"] = df_filtered.apply(
        lambda row: tuple(sorted([row["source"], row["target"]])), axis=1
    )
    agg_dict = {feat: "mean" for feat in all_features}
    agg_dict["accuracy"] = "mean"
    df_grouped = df_filtered.groupby("lang_pair").agg(agg_dict).reset_index()

    # Step 4: Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df_grouped[all_features]), columns=all_features)
    y = df_grouped["accuracy"].values

    # Step 5: Baseline with all features
    model = LinearRegression()
    y_pred = cross_val_predict(model, X_scaled.values, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
    baseline_adj_r2 = adjusted_r2_score(y, y_pred, p=len(all_features))

    print(f"\n=== Single-step Regression Ablation for {model_name} | {method_name} | {mode_name} ===")
    print(f"Baseline Adjusted R² (all features) = {baseline_adj_r2:.4f}\n")

    results = []

    # Step 6: Remove each feature once
    for feat in all_features:
        subset_feats = [f for f in all_features if f != feat]
        X_sub = X_scaled[subset_feats].values

        y_pred = cross_val_predict(model, X_sub, y, cv=KFold(n_splits=10, shuffle=True, random_state=42))
        adj_r2 = adjusted_r2_score(y, y_pred, p=len(subset_feats))

        delta_adj_r2 = baseline_adj_r2 - adj_r2

        print(f"Remove {feat:25s} -> ΔAdj R² = {delta_adj_r2:+.4f} (Adj R² = {adj_r2:.4f})")

        results.append({
            "model": model_name,
            "method": method_name,
            "mode": mode_name,
            "feature_removed": feat,
            "adj_r2_removed": adj_r2,
            "delta_adj_r2": delta_adj_r2
        })

    # Step 7: Save results
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, f"single_step_regression_{model_name}_{method_name}_{mode_name}.csv")
    pd.DataFrame(results).sort_values("delta_adj_r2", ascending=False).to_csv(out_file, index=False)

    print(f"\n[✓] Single-step regression results saved to: {out_file}")

# -----------------------------
# SHAP Analysis
# -----------------------------
def shap_analysis(merged_df, model_name, method_name, mode_name, output_path="results/transfer/"):
    # Step 1: Define candidate features
    if model_name != "GPT-4o-Mini-2024-07-18":
        all_features = [
            "phylogeny_dist_sym", "syntax_dist_sym", "geo_dist_sym",
            "log_data_prop_sum", "log_data_prop_min",
            "optimized", "eflomal_sym"
        ]
    else:
        all_features = [
            "phylogeny_dist_sym", "syntax_dist_sym", "geo_dist_sym",
            "log_data_prop_sum", "log_data_prop_min", "eflomal_sym"
        ]
    
    # Step 2: Filter data (remove source==target, drop NaNs)
    df_filtered = merged_df[merged_df["source"] != merged_df["target"]].copy()
    drop_cols = all_features + ["accuracy"]
    df_filtered = df_filtered.dropna(subset=drop_cols).copy()

    # Step 3: Symmetrize language pairs
    df_filtered["lang_pair"] = df_filtered.apply(
        lambda row: tuple(sorted([row["source"], row["target"]])), axis=1
    )
    agg_dict = {feat: "mean" for feat in all_features}
    agg_dict["accuracy"] = "mean"
    df_grouped = df_filtered.groupby("lang_pair").agg(agg_dict).reset_index()

    X = df_grouped[all_features].values
    y = df_grouped["accuracy"].values

    # Step 4: 5-Fold CV training + SHAP
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_shap_values = []
    all_X = []
    metrics = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        metrics.append({"MAE": mae})

        explainer = shap.TreeExplainer(model)
        shap_values_fold = explainer.shap_values(X_test)

        all_shap_values.append(shap_values_fold)
        all_X.append(X_test)
    
    shap_values_all = np.vstack(all_shap_values)
    X_all = np.vstack(all_X)

    # Step 5: Aggregate feature importance (mean |SHAP|)
    mean_abs_shap = np.abs(shap_values_all).mean(axis=0)
    shap_df = pd.DataFrame({
        "model": model_name,
        "method": method_name,
        "mode": mode_name,
        "feature": all_features,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    # Step 6: Save results
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, f"shap_importance_{model_name}_{method_name}_{mode_name}_5fold.csv")
    shap_df.to_csv(out_file, index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_summary = metrics_df.mean().to_frame("mean").T
    metrics_out = os.path.join(output_path, f"shap_metrics_{model_name}_{method_name}_{mode_name}_5fold.csv")
    metrics_summary.to_csv(metrics_out, index=False)

    print("\n=== SHAP Feature Importance (mean |SHAP|, 5-fold) ===")
    print(shap_df)
    print("\n=== CV Metrics (5-fold) ===")
    print(metrics_summary)
    print(f"\n[✓] SHAP results saved to: {out_file}")
    print(f"[✓] Metrics saved to: {metrics_out}")

    # Step 7: Summary plot
    try:
        shap.summary_plot(
            shap_values_all, 
            features=X_all, 
            feature_names=[FEATURE_NAME_MAP[f] for f in all_features], 
            show=False
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_path, f"shap_summary_{model_name}_{method_name}_{mode_name}_5fold.png"), 
            dpi=300
        )
        plt.close()
    except Exception as e:
        print(f"[!] SHAP summary plot failed: {e}")

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--method_name", type=str)
    parser.add_argument("--accuracy_table", type=str, default="./datasets/transfer_results.csv")
    args = parser.parse_args()

    args.language_features = f"./datasets/per_lang_stats_{args.model_name}.csv"
    args.pair_features = f"./datasets/per_pair_stats_{args.model_name}.csv"
    accuracy_df = pd.read_csv(args.accuracy_table)

    merged_df = preprocess_and_merge(args.language_features, args.pair_features, accuracy_df, model_name=args.model_name, method_name=args.method_name, mode_name=args.mode_name)
    cor_analysis(merged_df,model_name=args.model_name,method_name=args.method_name,mode_name=args.mode_name)
    exhaustive_feature_selection(merged_df, args.model_name, args.method_name, args.mode_name)
    single_step_regression(merged_df, args.model_name, args.method_name, args.mode_name)
    shap_analysis(merged_df, args.model_name, args.method_name, args.mode_name)