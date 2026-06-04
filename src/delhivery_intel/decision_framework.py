from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_route_type_policy(df: pd.DataFrame, hub_metrics: pd.DataFrame, penalty: float = 500.0) -> pd.DataFrame:
    # 1. Merge source facility graph position metrics
    merged = df.merge(
        hub_metrics.rename(columns={
            "hub": "source_facility",
            "betweenness": "source_betweenness",
            "in_degree": "source_in_degree",
            "out_degree": "source_out_degree",
            "clustering": "source_clustering",
        }),
        on="source_facility",
        how="left",
    )
    merged["source_betweenness"] = merged["source_betweenness"].fillna(0.0)
    merged["source_clustering"] = merged["source_clustering"].fillna(0.0)
    
    if "distance_km" not in merged.columns:
        merged["distance_km"] = 100.0

    # 2. Train an ML model (RandomForestClassifier) to predict SLA breach probability
    features = ["route_type", "time_bucket", "distance_km", "source_betweenness", "source_clustering"]
    if "predicted_eta" in merged.columns:
        features.append("predicted_eta")
    X = merged[features]
    y = merged["sla_breach"].values
    
    cat_cols = ["route_type", "time_bucket"]
    num_cols = ["distance_km", "source_betweenness", "source_clustering"]
    if "predicted_eta" in merged.columns:
        num_cols.append("predicted_eta")
    
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    
    model = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    train_mask = merged["predicted_eta"].notna()
    X_valid = X[train_mask]
    y_valid = y[train_mask]
    
    split_idx = int(len(X_valid) * 0.8)
    model.fit(X_valid.iloc[:split_idx], y_valid[:split_idx])
    
    # 3. Create corridor profiles
    profile_cols = ["source_facility", "dest_facility", "time_bucket", "distance_km", 
                       "source_betweenness", "source_clustering"]
    if "predicted_eta" in merged.columns:
        # Get mean predicted_eta for the profile
        profile_means = merged.groupby(["source_facility", "dest_facility", "time_bucket"])["predicted_eta"].mean().reset_index()
        profiles = merged[profile_cols].drop_duplicates()
        profiles = profiles.merge(profile_means, on=["source_facility", "dest_facility", "time_bucket"], how="left")
    else:
        profiles = merged[profile_cols].drop_duplicates()
    
    # Evaluate both FTL and Carting for every profile
    eval_ftl = profiles.copy()
    eval_ftl["route_type"] = "FTL"
    eval_ftl["transport_cost"] = eval_ftl["distance_km"] * 1.4
    
    eval_cart = profiles.copy()
    eval_cart["route_type"] = "Carting"
    eval_cart["transport_cost"] = eval_cart["distance_km"] * 1.0

    X_ftl = eval_ftl[features]
    X_cart = eval_cart[features]
    
    # Predict Probability of Breach
    eval_ftl["p_breach"] = model.predict_proba(X_ftl)[:, 1] if len(model.classes_) > 1 else 0.0
    eval_cart["p_breach"] = model.predict_proba(X_cart)[:, 1] if len(model.classes_) > 1 else 0.0
    
    # Expected Cost = Transport Cost + P(Breach) * Penalty
    eval_ftl["expected_cost"] = eval_ftl["transport_cost"] + (eval_ftl["p_breach"] * penalty)
    eval_cart["expected_cost"] = eval_cart["transport_cost"] + (eval_cart["p_breach"] * penalty)
    
    eval_ftl = eval_ftl.rename(columns={"expected_cost": "FTL_expected_cost"})
    eval_cart = eval_cart.rename(columns={"expected_cost": "Carting_expected_cost"})
    
    policy = pd.merge(eval_ftl, eval_cart[["source_facility", "dest_facility", "time_bucket", "Carting_expected_cost"]],
                      on=["source_facility", "dest_facility", "time_bucket"])
    
    policy["recommended_route_type"] = np.where(
        policy["FTL_expected_cost"] < policy["Carting_expected_cost"], "FTL", "Carting"
    )
    
    return policy[["source_facility", "dest_facility", "time_bucket", "distance_km",
                   "FTL_expected_cost", "Carting_expected_cost", "recommended_route_type"]]

