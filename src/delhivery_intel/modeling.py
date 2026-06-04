from __future__ import annotations

import numpy as np
import pandas as pd
from node2vec import Node2Vec
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr


def _within_15(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs(y_true - y_pred) / denom <= 0.15))


def train_baseline(df: pd.DataFrame) -> dict:
    features = ["route_type", "time_bucket", "osrm_eta_minutes", "hour_of_day", "day_of_week", "is_weekend", "is_night", "month"]
    if "distance_km" in df.columns:
        features.append("distance_km")

    df_sorted = df.sort_values("departure_ts").copy() if "departure_ts" in df.columns else df.copy()
    X = df_sorted[features]
    y = df_sorted["actual_transit_minutes"].values
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Fix 1: Dumb Baselines
    # Baseline 1: Median actual time per corridor
    corridor_medians = df_sorted.iloc[:split_idx].groupby("corridor")["actual_transit_minutes"].median()
    test_corridors = df_sorted.iloc[split_idx:]["corridor"]
    median_preds = test_corridors.map(corridor_medians).fillna(df_sorted.iloc[:split_idx]["actual_transit_minutes"].median()).values
    median_mae = float(mean_absolute_error(y_test, median_preds))
    median_rmse = float(np.sqrt(mean_squared_error(y_test, median_preds)))
    
    # Baseline 2: Linear Regression on distance + hour
    lr = LinearRegression()
    lr_features = ["hour_of_day"]
    if "distance_km" in X_train.columns:
        lr_features = ["distance_km", "hour_of_day"]
    lr.fit(X_train[lr_features], y_train)
    lr_preds = lr.predict(X_test[lr_features])
    lr_mae = float(mean_absolute_error(y_test, lr_preds))
    lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_preds)))
    
    print(f"Median Baseline MAE: {median_mae:.2f}, RMSE: {median_rmse:.2f}")
    print(f"LR Baseline MAE: {lr_mae:.2f}, RMSE: {lr_rmse:.2f}")

    cat_cols = ["route_type", "time_bucket"]
    num_cols = [c for c in features if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("scale", StandardScaler())]), num_cols),
        ]
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    rf_mae = float(mean_absolute_error(y_test, preds))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    
    df_sorted["predicted_eta"] = np.nan
    df_sorted.loc[df_sorted.index[split_idx:], "predicted_eta"] = preds
    
    rf_acc15 = _within_15(y_test, preds)
    median_acc15 = _within_15(y_test, median_preds)
    lr_acc15 = _within_15(y_test, lr_preds)
    print(f"Median Baseline Acc@15%: {median_acc15:.4f}")
    print(f"LR Baseline Acc@15%:     {lr_acc15:.4f}")
    print(f"RF Model Acc@15%:        {rf_acc15:.4f}")
    
    return {
        "model": pipe,
        "features": features,
        "df_with_preds": df_sorted,
        "mae": rf_mae,
        "rmse": rf_rmse,
        "acc15": _within_15(y_test, preds),
        "median_mae": median_mae,
        "median_rmse": median_rmse,
        "median_acc15": median_acc15,
        "lr_mae": lr_mae,
        "lr_rmse": lr_rmse,
        "lr_acc15": lr_acc15,
        "test_df": X_test.copy(),
        "y_test": y_test,
        "preds": preds,
    }


def _node_embeddings(df: pd.DataFrame, dim: int = 32) -> tuple[dict[str, np.ndarray], float]:
    import networkx as nx

    g = nx.DiGraph()
    if "delay_ratio" not in df.columns:
        df = df.copy()
        df["delay_ratio"] = (df["actual_transit_minutes"] / df["osrm_eta_minutes"].replace(0, np.nan)).clip(upper=5.0)

    weighted = (
        df.groupby(["source_facility", "dest_facility"])
        .agg(trips=("corridor", "count"), delay_weight=("delay_ratio", "median"))
        .reset_index()
    )
    weighted["weight"] = weighted["trips"] * weighted["delay_weight"]
    # Edge weight = trips × median_delay_ratio: biases Node2Vec walks toward corridors that are both high-volume and high-delay, encoding operationally relevant structural position.
    for _, row in weighted.iterrows():
        g.add_edge(str(row["source_facility"]), str(row["dest_facility"]), weight=float(row["weight"]))

    if g.number_of_nodes() < 2:
        return {}, 1.0

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        n2v_1 = Node2Vec(g, dimensions=dim, walk_length=80, num_walks=200, workers=1, weight_key="weight", seed=42)
        w2v_1 = n2v_1.fit(window=5, min_count=1)
        emb_1 = {node: w2v_1.wv[node] for node in g.nodes()}
        
        n2v_2 = Node2Vec(g, dimensions=dim, walk_length=80, num_walks=200, workers=1, weight_key="weight", seed=100)
        w2v_2 = n2v_2.fit(window=5, min_count=1)
        emb_2 = {node: w2v_2.wv[node] for node in g.nodes()}
    
    nodes = list(g.nodes())
    rng = np.random.default_rng(42)
    sample_nodes = rng.choice(nodes, size=min(len(nodes), 200), replace=False)
    
    dists_1 = []
    dists_2 = []
    for i in range(len(sample_nodes)):
        for j in range(i + 1, len(sample_nodes)):
            n1, n2 = sample_nodes[i], sample_nodes[j]
            dists_1.append(1 - cosine(emb_1[n1], emb_1[n2]))
            dists_2.append(1 - cosine(emb_2[n1], emb_2[n2]))
            
    # Add a tiny epsilon to avoid zero variance if embeddings are identical
    dists_1 = np.array(dists_1) + rng.normal(0, 1e-8, len(dists_1))
    dists_2 = np.array(dists_2) + rng.normal(0, 1e-8, len(dists_2))
    
    corr, _ = pearsonr(dists_1, dists_2)
    mean_sim = float(corr)
    
    if mean_sim < 0.65:
        print(f"WARNING: Node2Vec embeddings are unstable. Mean cosine similarity between runs is {mean_sim:.3f} (< 0.65). Results may not be reliable.")
        # Threshold set to 0.65: for a directed logistics network with seasonal variation, inter-run cosine similarity >0.65 indicates stable structural hierarchy. Score is reported as a diagnostic metric; pipeline continues regardless.
        
    return emb_1, mean_sim


def train_graph_enhanced(df: pd.DataFrame) -> dict:
    df_sorted = df.sort_values("departure_ts").copy() if "departure_ts" in df.columns else df.copy()
    split_idx_raw = int(len(df_sorted) * 0.8)   # must be after sort
    train_df_raw = df_sorted.iloc[:split_idx_raw]
    emb, mean_sim = _node_embeddings(train_df_raw, dim=32)
    tmp = df_sorted.copy()
    for i in range(32):
        tmp[f"src_emb_{i}"] = tmp["source_facility"].astype(str).map(lambda x: emb.get(x, np.zeros(32))[i])
        tmp[f"dst_emb_{i}"] = tmp["dest_facility"].astype(str).map(lambda x: emb.get(x, np.zeros(32))[i])

    unseen_src = tmp["source_facility"].astype(str).map(lambda x: x not in emb).sum()
    unseen_dst = tmp["dest_facility"].astype(str).map(lambda x: x not in emb).sum()
    if unseen_src + unseen_dst > 0:
        print(f"[modeling] WARNING: {unseen_src} source and {unseen_dst} dest nodes had no embedding (zero-vector fallback used).")

    features = ["route_type", "time_bucket", "osrm_eta_minutes", "hour_of_day", "day_of_week", "is_weekend", "is_night", "month"] + \
               [f"src_emb_{i}" for i in range(32)] + [f"dst_emb_{i}" for i in range(32)]
    
    if "distance_km" in tmp.columns:
        features.append("distance_km")

    X = tmp[features]
    y = tmp["actual_transit_minutes"].values
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y[:split_idx], y[split_idx:]

    cat_cols = ["route_type", "time_bucket"]
    num_cols = [c for c in features if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    # Hyperparameters intentionally identical to baseline so that MAE difference is attributable only to embedding features, not model capacity.
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    rf_mae = float(mean_absolute_error(y_test, preds))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    
    return {
        "model": pipe,
        "mae": rf_mae,
        "rmse": rf_rmse,
        "acc15": _within_15(y_test, preds),
        "embedding_stability": mean_sim
    }

