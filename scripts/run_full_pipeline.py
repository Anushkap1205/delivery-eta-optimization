from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from delhivery_intel.config import DATA_PROCESSED, DATA_RAW, FIGURES, METRICS, REPORTS, TABLES, ensure_dirs
from delhivery_intel.decision_framework import build_route_type_policy
from delhivery_intel.graph_ops import build_corridor_aggregates, build_network_graph, compute_hub_metrics
from delhivery_intel.modeling import train_baseline, train_graph_enhanced
from delhivery_intel.pipeline import load_and_prepare
from delhivery_intel.reporting import estimate_upgrade_impact, render_strategy_memo, top_bottleneck_hubs
from delhivery_intel.visualization import plot_delay_corridors, plot_top_hubs, plot_network_topology


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_RAW / "trips.csv"),
        help="Path to input CSV file.",
    )
    args = parser.parse_args()
    input_file = Path(args.input)
    if not input_file.exists():
        raise FileNotFoundError(
            f"Expected input file at {input_file}. Please place your trip data there."
        )

    df = load_and_prepare(str(input_file))
    df.to_csv(DATA_PROCESSED / "trips_prepared.csv", index=False)

    edge_df = build_corridor_aggregates(df)
    edge_df.to_csv(TABLES / "corridor_aggregates.csv", index=False)

    graph = build_network_graph(edge_df)
    hub_metrics = compute_hub_metrics(graph)
    hub_metrics.to_csv(TABLES / "hub_metrics.csv", index=False)

    chronic_corridors = edge_df[edge_df["chronic_delay"]].copy()
    chronic_corridors["sla_breach_contribution"] = chronic_corridors["trips"] * chronic_corridors["breach_rate"]
    top_corridors = chronic_corridors.sort_values("sla_breach_contribution", ascending=False).head(10)
    top_corridors.to_csv(TABLES / "top_delay_corridors.csv", index=False)

    baseline = train_baseline(df)
    graph_model = train_graph_enhanced(df)

    with open(METRICS / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_mae": baseline["mae"],
                "baseline_rmse": baseline["rmse"],
                "graph_mae": graph_model["mae"],
                "graph_rmse": graph_model["rmse"],
                "median_mae": baseline["median_mae"],
                "median_rmse": baseline["median_rmse"],
                "lr_mae": baseline["lr_mae"],
                "lr_rmse": baseline["lr_rmse"],
                "baseline_acc15": baseline["acc15"],
                "graph_acc15": graph_model["acc15"],
                "graph_advantage_mae_reduction": baseline["mae"] - graph_model["mae"],
                "graph_advantage_acc15_delta": graph_model["acc15"] - baseline["acc15"],
                "embedding_stability": graph_model["embedding_stability"]
            },
            f,
            indent=2,
        )

    policy = build_route_type_policy(baseline["df_with_preds"], hub_metrics)
    policy.to_csv(TABLES / "ftl_vs_carting_policy.csv", index=False)

    top_hubs_df = top_bottleneck_hubs(hub_metrics, top_n=5)
    top_hubs_df.to_csv(TABLES / "top_5_bottleneck_hubs.csv", index=False)
    plot_top_hubs(top_hubs_df, str(FIGURES / "top_bottleneck_hubs.png"))
    plot_delay_corridors(top_corridors, str(FIGURES / "top_delay_corridors.png"))
    plot_network_topology(graph, top_hubs_df, top_corridors, str(FIGURES / "network_topology.png"))
    top3 = list(top_hubs_df["hub"].head(3))
    impact = estimate_upgrade_impact(df, top3)

    memo_path = REPORTS / "network_operations_strategy_memo.md"
    render_strategy_memo(
        out_path=str(memo_path),
        top_hubs=top_hubs_df,
        top_corridors=top_corridors.head(5),
        baseline_metrics=baseline,
        graph_metrics=graph_model,
        impact=impact,
    )

    print("Pipeline completed.")
    print(f"Memo generated at: {memo_path}")


if __name__ == "__main__":
    main()

