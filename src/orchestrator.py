"""
orchestrator.py — LangGraph Orchestrator
Industrial AI Predictive Maintenance | BearingMind

Defines a directed graph with conditional routing. The system decides
its own next step based on intermediate results — this is what makes
BearingMind agentic.

Graph:
    START → detect → [anomaly?] → explain → diagnose → alert → END
                        │ no
                        └──→ log_healthy → END

Usage:
    python orchestrator.py ../results/feature_matrix.csv ../results 950
    python orchestrator.py ../results/feature_matrix.csv ../results 100
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import TypedDict
from pathlib import Path

from langgraph.graph import StateGraph, END


# ── State schema ──────────────────────────────────────────────────────────────

class BearingMindState(TypedDict, total=False):
    snapshot_index: int
    anomaly_scores: dict
    rul_scores: dict
    is_anomaly: bool
    worst_bearing: str
    shap_context: dict
    fault_type: str
    rca_report: dict
    urgency: str
    alert_result: dict
    path_taken: list
    skipped_reason: str


# ── Conditional routing ───────────────────────────────────────────────────────

def should_investigate(state: BearingMindState) -> str:
    """After detection: investigate or skip?"""
    if state.get("is_anomaly"):
        return "explain"
    return "log_healthy"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(feature_matrix: pd.DataFrame,
                anomaly_det,
                rul_pred,
                shap_explainer,
                rca_agent,
                alert_agent):
    """
    Build the LangGraph with conditional routing.

    Uses closures to capture model/agent references — LangGraph state
    only carries serializable data (scores, flags, reports), not
    model objects.
    """

    # ── Node: detect ──────────────────────────────────────────────────
    def detect_node(state: BearingMindState) -> dict:
        snapshot_idx = state["snapshot_index"]
        print(f"\n[detect] Scoring snapshot {snapshot_idx} ...")

        scores_df = anomaly_det.score_all()
        anomaly_scores = {}
        is_anomaly = False
        worst_score = -np.inf
        worst_bearing = None

        for bid in anomaly_det.bearing_ids_:
            score_col = f"{bid}_score"
            flag_col  = f"{bid}_flag"
            if score_col in scores_df.columns:
                score = float(scores_df[score_col].iloc[snapshot_idx])
                flag  = int(scores_df[flag_col].iloc[snapshot_idx])
                anomaly_scores[bid] = score
                if flag == 1:
                    is_anomaly = True
                if score > worst_score:
                    worst_score = score
                    worst_bearing = bid

        rul_scores = {}
        for bid in rul_pred.bearing_ids_:
            preds = rul_pred.predictors_[bid].predict(feature_matrix)
            rul_scores[bid] = float(preds[snapshot_idx])
            if rul_scores[bid] <= 0.25:
                is_anomaly = True

        print(f"  Worst bearing: {worst_bearing} "
              f"(score: {worst_score:.4f})")
        print(f"  RUL: {', '.join(f'{b}={v:.3f}' for b, v in rul_scores.items())}")
        print(f"  Anomaly detected: {is_anomaly}")

        return {
            "anomaly_scores": anomaly_scores,
            "rul_scores":     rul_scores,
            "is_anomaly":     is_anomaly,
            "worst_bearing":  worst_bearing,
            "path_taken":     state.get("path_taken", []) + ["detect"],
        }

    # ── Node: log_healthy ─────────────────────────────────────────────
    def log_healthy_node(state: BearingMindState) -> dict:
        snap = state["snapshot_index"]
        print(f"\n[log_healthy] Snapshot {snap} — no anomaly. "
              f"Skipping SHAP/RCA/Alert.")
        return {
            "skipped_reason": "No anomaly detected — all bearings healthy.",
            "urgency": "LOW",
            "path_taken": state.get("path_taken", []) + ["log_healthy"],
        }

    # ── Node: explain (SHAP) ──────────────────────────────────────────
    def explain_node(state: BearingMindState) -> dict:
        snapshot_idx = state["snapshot_index"]
        print(f"\n[explain] Running SHAP on snapshot {snapshot_idx} ...")

        shap_result = shap_explainer.explain_snapshot(snapshot_idx)
        worst = shap_result.get(
            "most_anomalous_bearing", state["worst_bearing"])
        fault_type = "undetermined"
        if worst and worst in shap_result:
            fault_type = shap_result[worst].get(
                "probable_fault", "undetermined")

        print(f"  Most anomalous: {worst}")
        print(f"  Fault type: {fault_type}")

        return {
            "shap_context": shap_result,
            "fault_type":   fault_type,
            "path_taken":   state.get("path_taken", []) + ["explain"],
        }

    # ── Node: diagnose (RCA + MCP) ────────────────────────────────────
    def diagnose_node(state: BearingMindState) -> dict:
        worst = state.get("worst_bearing", "unknown")
        print(f"\n[diagnose] Running RCA agent for {worst} ...")

        target_context = state["shap_context"].get(
            worst, state["shap_context"])
        report = rca_agent.analyze(target_context, bearing_id=worst)

        return {
            "rca_report": report,
            "urgency":    report.get("urgency", "MEDIUM"),
            "path_taken": state.get("path_taken", []) + ["diagnose"],
        }

    # ── Node: alert ───────────────────────────────────────────────────
    def alert_node(state: BearingMindState) -> dict:
        print(f"\n[alert] Routing {state.get('urgency', 'MEDIUM')} alert ...")
        result = alert_agent.process(state["rca_report"])
        return {
            "alert_result": result,
            "path_taken":   state.get("path_taken", []) + ["alert"],
        }

    # ── Wire the graph ────────────────────────────────────────────────
    graph = StateGraph(BearingMindState)

    graph.add_node("detect",      detect_node)
    graph.add_node("log_healthy", log_healthy_node)
    graph.add_node("explain",     explain_node)
    graph.add_node("diagnose",    diagnose_node)
    graph.add_node("alert",       alert_node)

    graph.set_entry_point("detect")
    graph.add_conditional_edges(
        "detect", should_investigate,
        {"explain": "explain", "log_healthy": "log_healthy"})
    graph.add_edge("explain", "diagnose")
    graph.add_edge("diagnose", "alert")
    graph.add_edge("alert", END)
    graph.add_edge("log_healthy", END)

    return graph.compile()


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(feature_matrix_path: str,
                 models_dir: str,
                 snapshot_index: int = None,
                 output_dir: str = "results/rca",
                 api_key: str = None) -> dict:
    """Run the complete BearingMind agentic pipeline."""

    sys.path.insert(0, os.path.dirname(__file__))
    from isolation_forest import BearingAnomalyDetector, SingleBearingDetector
    from rul_lstm import BearingRULPredictor, SingleBearingRUL, LSTMRULModel
    from shap_explainer import BearingShapExplainer
    from mcp_equipment_manual import EquipmentManualMCP
    from mcp_cmms import CMMSMCP
    from rca_agent import RCAAgent
    from alert_agent import AlertAgent

    import __main__
    for cls in [BearingAnomalyDetector, SingleBearingDetector,
                BearingRULPredictor, SingleBearingRUL, LSTMRULModel]:
        setattr(__main__, cls.__name__, cls)

    print("=" * 60)
    print("BearingMind — Agentic Pipeline (LangGraph)")
    print("=" * 60)

    # ── Load data + models ────────────────────────────────────────────
    print("\n[setup] Loading feature matrix ...")
    df = pd.read_csv(feature_matrix_path, index_col=0)
    if snapshot_index is None:
        snapshot_index = len(df) - 1
    print(f"  {df.shape[0]} snapshots, analyzing #{snapshot_index}")

    print("[setup] Loading ML models ...")
    if_dir  = os.path.join(models_dir, "if", "models")
    rul_dir = os.path.join(models_dir, "rul", "models")

    anomaly_det = BearingAnomalyDetector()
    anomaly_det.fit_from_df(df)
    if os.path.isdir(if_dir):
        anomaly_det.load_models(if_dir)

    rul_pred = BearingRULPredictor()
    if os.path.isdir(rul_dir):
        rul_pred.feature_matrix_ = df
        rul_pred.load_models(rul_dir)
        rul_pred.bearing_ids_ = sorted(rul_pred.predictors_.keys())

    print("[setup] Initializing SHAP explainer ...")
    shap_exp = BearingShapExplainer(anomaly_det, rul_pred)
    shap_exp.fit(df=df, n_background=100)

    # ── MCP servers ───────────────────────────────────────────────────
    print("[setup] Initializing MCP servers ...")
    manual_mcp = EquipmentManualMCP()
    manual_mcp.load()

    cmms_db = os.path.join(output_dir, "cmms.db")
    cmms_mcp = CMMSMCP(db_path=cmms_db)
    cmms_mcp.initialize()

    weather_mcp = None
    try:
        from mcp_weather import WeatherMCP
        weather_mcp = WeatherMCP()
        weather_mcp.fetch()
        print("  Weather MCP: connected")
    except Exception as e:
        print(f"  Weather MCP: skipped ({e})")

    # ── Agents ────────────────────────────────────────────────────────
    rca = RCAAgent(
        manual_mcp=manual_mcp, cmms_mcp=cmms_mcp,
        weather_mcp=weather_mcp, api_key=api_key)
    alert = AlertAgent(
        cmms_mcp=cmms_mcp,
        log_path=os.path.join(output_dir, "alert_log.json"))

    # ── Build and run graph ───────────────────────────────────────────
    print("\n[setup] Building LangGraph orchestrator ...")
    app = build_graph(df, anomaly_det, rul_pred, shap_exp, rca, alert)

    print(f"\n{'='*60}")
    print(f"Running graph for snapshot {snapshot_index} ...")
    print(f"{'='*60}")

    result = app.invoke({"snapshot_index": snapshot_index, "path_taken": []})

    # ── Save outputs ──────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if result.get("rca_report"):
        report_path = os.path.join(output_dir, "rca_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(result["rca_report"].get("report_text", ""))
        print(f"\n  Report saved → {report_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Pipeline complete")
    print(f"  Path: {' → '.join(result.get('path_taken', []))}")
    print(f"  Urgency: {result.get('urgency', 'N/A')}")
    if result.get("skipped_reason"):
        print(f"  Skipped: {result['skipped_reason']}")
    if result.get("alert_result"):
        print(f"  Alert: {result['alert_result'].get('summary', '')}")
    print(f"{'='*60}")

    cmms_mcp.close()
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python orchestrator.py <feature_matrix.csv> "
              "<models_dir> [snapshot_index] [output_dir]")
        print("\nExamples:")
        print("  python orchestrator.py ../results/feature_matrix.csv "
              "../results 950  (anomaly → full pipeline)")
        print("  python orchestrator.py ../results/feature_matrix.csv "
              "../results 100  (healthy → skips SHAP/RCA)")
        sys.exit(1)

    result = run_pipeline(
        feature_matrix_path=sys.argv[1],
        models_dir=sys.argv[2],
        snapshot_index=int(sys.argv[3]) if len(sys.argv) > 3 else None,
        output_dir=sys.argv[4] if len(sys.argv) > 4 else "results/rca",
    )