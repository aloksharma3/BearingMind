"""
rca_agent.py — Root Cause Analysis Agent
Industrial AI Predictive Maintenance | BearingMind

The RCA agent is the "brain" that turns ML outputs into actionable
maintenance decisions. It receives SHAP evidence, queries equipment
manuals and CMMS history via MCP, and produces a structured fault report
that a maintenance engineer can act on.

What this does:
    1. Receives SHAP context from shap_explainer.py
       ("kurtosis and BPFI are the top contributors → inner race fault")
    2. Calls Equipment Manual MCP → "what does the manual say about
       inner race faults?"
    3. Calls CMMS MCP → "when was the last work order? what parts are
       in stock?"
    4. Synthesizes everything into a structured RCA report using Claude API
    5. Returns a report with: diagnosis, evidence, root causes,
       recommended actions, parts needed, urgency level

Why a separate agent (not just prompt engineering):
    - The RCA agent has its own TOOLS (MCP servers) that it calls
    - It makes decisions about WHAT to query based on the fault type
    - It can be swapped independently (Claude → GPT → Llama) without
      touching the ML pipeline
    - In LangGraph, it's a node with defined inputs/outputs

Architecture:
    ┌────────────────┐     ┌─────────────────────┐
    │ SHAP Explainer │────▶│     RCA Agent        │
    │ (evidence)     │     │                      │
    └────────────────┘     │  1. Parse SHAP       │
                           │  2. Query Manual MCP │──▶ Equipment Manual MCP
                           │  3. Query CMMS MCP   │──▶ CMMS MCP (SQLite)
                           │  4. Call Claude API   │
                           │  5. Return report     │
                           └──────────┬────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │  Structured RCA       │
                           │  Report (dict + text) │
                           └──────────────────────┘

Usage:
    from src.rca_agent import RCAAgent
    from src.mcp_equipment_manual import EquipmentManualMCP
    from src.mcp_cmms import CMMSMCP

    # Initialize MCP servers
    manual_mcp = EquipmentManualMCP()
    manual_mcp.load()
    cmms_mcp = CMMSMCP()
    cmms_mcp.initialize()

    # Initialize RCA agent with both tools
    rca = RCAAgent(
        manual_mcp=manual_mcp,
        cmms_mcp=cmms_mcp,
        api_key="sk-ant-..."       # or set ANTHROPIC_API_KEY env var
    )

    # Generate RCA report from SHAP context
    shap_context = explainer.explain_snapshot(950)
    report = rca.analyze(shap_context)
    print(report["report_text"])
"""

import os
import json
from datetime import datetime
from typing import Optional
from pathlib import Path

# ── Optional: Claude API ──────────────────────────────────────────────────────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ── Fault type → manual query mapping ─────────────────────────────────────────
# Maps SHAP-inferred fault types to the queries we send to the manual MCP.
# This is how the agent decides WHAT to look up based on ML evidence.

FAULT_QUERY_MAP = {
    "outer race fault": [
        "outer race BPFO fault diagnosis and maintenance",
        "bearing housing inspection outer race defect",
    ],
    "inner race fault": [
        "inner race BPFI fault diagnosis maintenance actions",
        "bearing lubrication inner race spalling",
    ],
    "rolling element (ball) fault": [
        "ball fault BSF rolling element detection",
        "lubricant contamination electrical discharge bearing",
    ],
    "impulsive fault (spalling / cracking)": [
        "kurtosis impulsive bearing spalling early damage",
        "bearing replacement procedure spalling",
    ],
    "impulsive fault (early stage)": [
        "crest factor early stage bearing damage detection",
        "lubrication schedule preventive maintenance",
    ],
    "surface degradation (broadband wear)": [
        "high frequency energy surface degradation wear",
        "bearing failure stages catastrophic",
    ],
    "distributed damage (multiple fault modes)": [
        "spectral entropy distributed damage multiple fault",
        "bearing failure model stages severity",
    ],
    "early-stage damage": [
        "margin factor early stage damage detection",
        "preventive maintenance schedule bearing",
    ],
}


# ── RCA System Prompt ─────────────────────────────────────────────────────────

RCA_SYSTEM_PROMPT = """You are an expert Root Cause Analysis agent for industrial rotating equipment.
You work for a plant maintenance team and your job is to analyze bearing fault
evidence from ML models and produce actionable maintenance recommendations.

You receive three types of information:
1. SHAP EVIDENCE — which vibration features are driving the anomaly/RUL prediction
2. EQUIPMENT MANUAL CONTEXT — relevant passages from maintenance manuals
3. CMMS HISTORY — work order history, asset info, and spare parts availability

Your output must be a structured RCA report with these sections:

## DIAGNOSIS
One-sentence summary of the detected fault condition.

## EVIDENCE
Bullet list of the ML evidence (SHAP features, anomaly score, RUL estimate).
Cite specific numbers.

## PROBABLE ROOT CAUSE
Based on the manual context and SHAP evidence, explain the most likely
root cause. Reference specific manual guidance.

## MAINTENANCE HISTORY CONTEXT
What does the work order history tell us? When was last maintenance?
Any relevant patterns?

## RECOMMENDED ACTIONS
Numbered list of specific actions, in priority order.
Include: immediate actions, parts needed, timeline.

## URGENCY LEVEL
One of: CRITICAL (immediate shutdown), HIGH (schedule within 48 hours),
MEDIUM (next planned maintenance window), LOW (monitor and trend).

## PARTS REQUIRED
List specific part numbers and availability from inventory.

Keep your language precise and technical — this report goes to a maintenance
engineer, not a manager. Use bearing-specific terminology. Do not hedge
excessively — give clear recommendations based on the evidence available."""


# ── RCA Agent ─────────────────────────────────────────────────────────────────

class RCAAgent:
    """
    Root Cause Analysis agent with MCP tool access.

    Consumes SHAP evidence, queries equipment manuals and CMMS,
    and produces structured maintenance reports.

    Can operate in two modes:
        1. WITH Claude API → full LLM-generated report
        2. WITHOUT Claude API → template-based report (still useful,
           still demonstrates the MCP integration)

    Args:
        manual_mcp  : initialized EquipmentManualMCP instance
        cmms_mcp    : initialized CMMSMCP instance
        api_key     : Anthropic API key (or set ANTHROPIC_API_KEY env var)
        model       : Claude model to use (default: claude-sonnet-4-20250514)
    """

    def __init__(self,
                 manual_mcp,
                 cmms_mcp,
                 api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-20250514"):
        self.manual_mcp = manual_mcp
        self.cmms_mcp   = cmms_mcp
        self.model      = model

        # Initialize Claude client if available
        self.client_ = None
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if ANTHROPIC_AVAILABLE and api_key:
            self.client_ = anthropic.Anthropic(api_key=api_key)
            print("  RCA Agent: Claude API connected")
        else:
            print("  RCA Agent: template mode (no API key — set "
                  "ANTHROPIC_API_KEY for LLM-generated reports)")

    # ── Main analysis method ──────────────────────────────────────────────

    def analyze(self, shap_context: dict,
                bearing_id: str = None) -> dict:
        """
        Run full RCA pipeline for a bearing at a given snapshot.

        Steps:
            1. Extract fault type and evidence from SHAP context
            2. Query Equipment Manual MCP for relevant maintenance info
            3. Query CMMS MCP for work order history and parts
            4. Synthesize into RCA report (LLM or template)

        Args:
            shap_context : output from BearingShapExplainer.explain_snapshot()
                           or SingleBearingShapExplainer.explain_snapshot()
            bearing_id   : override bearing ID (auto-detected from context)

        Returns:
            dict with:
                - report_text      : formatted RCA report (str)
                - diagnosis         : one-line fault summary
                - urgency           : CRITICAL / HIGH / MEDIUM / LOW
                - manual_results    : raw manual query results
                - cmms_summary      : raw CMMS data
                - recommended_actions : list of action strings
                - mcp_calls_made    : log of MCP tool invocations
        """
        mcp_log = []

        # ── Step 1: Extract evidence from SHAP context ───────────────
        bearing_id = bearing_id or self._extract_bearing_id(shap_context)
        fault_type = self._extract_fault_type(shap_context)
        rca_text   = self._extract_rca_context(shap_context)

        print(f"\n{'='*60}")
        print(f"RCA Agent — Analyzing {bearing_id}")
        print(f"Detected fault type: {fault_type}")
        print(f"{'='*60}")

        # ── Step 2: Query Equipment Manual MCP ────────────────────────
        print("\n[1/3] Querying Equipment Manual MCP ...")
        manual_results = self._query_manual(fault_type, mcp_log)

        # ── Step 3: Query CMMS MCP ────────────────────────────────────
        print("[2/3] Querying CMMS MCP ...")
        cmms_summary = self._query_cmms(bearing_id, mcp_log)

        # ── Step 4: Generate RCA report ───────────────────────────────
        print("[3/3] Generating RCA report ...")
        if self.client_:
            report = self._generate_llm_report(
                rca_text, manual_results, cmms_summary, fault_type
            )
        else:
            report = self._generate_template_report(
                rca_text, manual_results, cmms_summary,
                fault_type, bearing_id
            )

        report["manual_results"]  = manual_results
        report["cmms_summary"]    = cmms_summary
        report["mcp_calls_made"]  = mcp_log
        report["bearing_id"]      = bearing_id
        report["fault_type"]      = fault_type

        print(f"\n✓ RCA report generated — urgency: {report['urgency']}")
        return report

    # ── MCP query methods ─────────────────────────────────────────────────

    def _query_manual(self, fault_type: str,
                      mcp_log: list) -> list[dict]:
        """Query Equipment Manual MCP based on detected fault type."""
        queries = FAULT_QUERY_MAP.get(fault_type, [
            f"{fault_type} bearing diagnosis",
            "bearing fault general maintenance procedure",
        ])

        all_results = []
        seen_sections = set()

        for query in queries:
            mcp_log.append({
                "tool": "query_manual",
                "query": query,
                "timestamp": datetime.now().isoformat(),
            })
            results = self.manual_mcp.query_manual(query, top_k=2)
            for r in results:
                # Deduplicate by section
                if r["section"] not in seen_sections:
                    all_results.append(r)
                    seen_sections.add(r["section"])
                    print(f"  → {r['source']}: {r['section']} "
                          f"(score: {r['relevance_score']:.3f})")

        return all_results

    def _query_cmms(self, bearing_id: str,
                    mcp_log: list) -> dict:
        """Query CMMS MCP for maintenance summary."""
        mcp_log.append({
            "tool": "get_maintenance_summary",
            "asset_id": bearing_id,
            "timestamp": datetime.now().isoformat(),
        })
        summary = self.cmms_mcp.get_maintenance_summary(bearing_id)

        # Log key findings
        days = summary.get("days_since_last_wo")
        n_wo = len(summary.get("work_orders", []))
        n_parts = len([p for p in summary.get("spare_parts", [])
                       if "error" not in p])
        print(f"  → {n_wo} work orders, {n_parts} compatible parts, "
              f"{days} days since last WO")

        return summary

    # ── LLM report generation ─────────────────────────────────────────────

    def _generate_llm_report(self, rca_text: str,
                              manual_results: list,
                              cmms_summary: dict,
                              fault_type: str) -> dict:
        """Generate RCA report using Claude API."""
        # Build the user prompt with all context
        manual_context = "\n\n".join(
            f"[{r['source']} — {r['section']}]\n{r['text']}"
            for r in manual_results
        )

        cmms_context = cmms_summary.get("summary_text", "No CMMS data.")

        user_prompt = f"""Analyze this bearing fault and produce an RCA report.

--- SHAP EVIDENCE ---
{rca_text}

--- EQUIPMENT MANUAL CONTEXT ---
{manual_context}

--- CMMS MAINTENANCE HISTORY ---
{cmms_context}

--- DETECTED FAULT TYPE ---
{fault_type}

Produce the structured RCA report as specified in your instructions."""

        try:
            response = self.client_.messages.create(
                model=self.model,
                max_tokens=2000,
                system=RCA_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            report_text = response.content[0].text

            # Parse urgency from report
            urgency = "MEDIUM"  # default
            for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                if f"## URGENCY LEVEL\n{level}" in report_text or \
                   f"URGENCY LEVEL\n{level}" in report_text or \
                   f"Urgency: {level}" in report_text:
                    urgency = level
                    break

            # Extract recommended actions (best-effort parse)
            actions = []
            in_actions = False
            for line in report_text.split("\n"):
                if "RECOMMENDED ACTIONS" in line:
                    in_actions = True
                    continue
                if in_actions and line.strip().startswith(("1.", "2.", "3.",
                                                           "4.", "5.")):
                    actions.append(line.strip())
                if in_actions and line.startswith("##") and \
                   "RECOMMENDED" not in line:
                    in_actions = False

            return {
                "report_text":        report_text,
                "diagnosis":          fault_type,
                "urgency":            urgency,
                "recommended_actions": actions,
                "generation_mode":    "llm",
            }

        except Exception as e:
            print(f"  ⚠ Claude API error: {e}")
            print("  Falling back to template mode ...")
            return self._generate_template_report(
                rca_text, manual_results, cmms_summary,
                fault_type, "unknown"
            )

    # ── Template report generation (no API needed) ────────────────────────

    def _generate_template_report(self, rca_text: str,
                                    manual_results: list,
                                    cmms_summary: dict,
                                    fault_type: str,
                                    bearing_id: str) -> dict:
        """
        Generate RCA report using structured templates.

        This mode works without any API key — proving the MCP integration
        works end-to-end. The template approach still produces a useful
        report by combining SHAP evidence with manual and CMMS context.
        """
        # Determine urgency from SHAP context
        urgency = self._assess_urgency(rca_text)

        # Extract top features from rca_text
        evidence_lines = []
        for line in rca_text.split("\n"):
            if line.strip().startswith("- ") or \
               line.strip().startswith("Anomaly score") or \
               line.strip().startswith("RUL score"):
                evidence_lines.append(line.strip())

        # Build manual context summary
        manual_summary = ""
        for r in manual_results[:3]:
            manual_summary += (
                f"\nPer {r['source']} ({r['section']}):\n"
                f"  {r['text'][:300]}...\n"
            )

        # Build CMMS context
        cmms_text = cmms_summary.get("summary_text", "No CMMS data available.")
        days_since = cmms_summary.get("days_since_last_wo", "unknown")

        # Recent work order findings
        wo_findings = ""
        for wo in cmms_summary.get("work_orders", [])[:2]:
            wo_findings += (
                f"\n  - {wo['wo_number']} ({wo['completed_date']}): "
                f"{wo['findings'][:150]}"
            )

        # Parts availability
        parts_lines = []
        for part in cmms_summary.get("spare_parts", []):
            if "error" not in part:
                status = "IN STOCK" if part.get("in_stock") else "OUT OF STOCK"
                parts_lines.append(
                    f"  - {part['part_number']}: {part['description']} — "
                    f"{status} ({part.get('qty_available', 0)} available)"
                )

        # Recommended actions based on fault type
        actions = self._get_recommended_actions(fault_type, urgency, cmms_summary)

        # Assemble report
        report_text = f"""
{'='*60}
ROOT CAUSE ANALYSIS REPORT — BearingMind
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Asset: {bearing_id}

## DIAGNOSIS
{fault_type.upper()} detected on {bearing_id} based on multi-model
SHAP analysis (Isolation Forest anomaly detection + LSTM RUL prediction).

## EVIDENCE (SHAP Analysis)
{rca_text}

## PROBABLE ROOT CAUSE
{manual_summary}

## MAINTENANCE HISTORY CONTEXT
Days since last maintenance: {days_since}
Recent work order findings:{wo_findings if wo_findings else ' No recent work orders.'}

## RECOMMENDED ACTIONS
{chr(10).join(f'{i+1}. {a}' for i, a in enumerate(actions))}

## URGENCY LEVEL
{urgency}

## PARTS REQUIRED
{chr(10).join(parts_lines) if parts_lines else '  No compatible parts found in inventory.'}

{'='*60}
Report generated by BearingMind RCA Agent (template mode)
MCP tools used: Equipment Manual MCP, CMMS MCP (SQLite)
{'='*60}
""".strip()

        return {
            "report_text":         report_text,
            "diagnosis":           fault_type,
            "urgency":             urgency,
            "recommended_actions": actions,
            "generation_mode":     "template",
        }

    # ── Helper methods ────────────────────────────────────────────────────

    def _extract_bearing_id(self, shap_context: dict) -> str:
        """Extract bearing_id from SHAP context."""
        # Direct field
        if "bearing_id" in shap_context:
            return shap_context["bearing_id"]
        # From most_anomalous_bearing (multi-bearing context)
        if "most_anomalous_bearing" in shap_context:
            return shap_context["most_anomalous_bearing"]
        return "unknown"

    def _extract_fault_type(self, shap_context: dict) -> str:
        """Extract fault type from SHAP context."""
        if "probable_fault" in shap_context:
            return shap_context["probable_fault"]
        # Check nested bearing contexts
        for key in ["b1_ch1", "b2_ch1", "b3_ch1", "b4_ch1"]:
            if key in shap_context and "probable_fault" in shap_context[key]:
                return shap_context[key]["probable_fault"]
        return "undetermined"

    def _extract_rca_context(self, shap_context: dict) -> str:
        """Extract RCA context string from SHAP context."""
        if "rca_context" in shap_context:
            return shap_context["rca_context"]
        if "system_rca_context" in shap_context:
            return shap_context["system_rca_context"]
        # From worst bearing
        worst = shap_context.get("most_anomalous_bearing")
        if worst and worst in shap_context:
            return shap_context[worst].get("rca_context", str(shap_context))
        return str(shap_context)

    def _assess_urgency(self, rca_text: str) -> str:
        """Determine urgency level from SHAP evidence text."""
        text_lower = rca_text.lower()
        if "critical" in text_lower:
            return "CRITICAL"
        if "warning" in text_lower:
            return "HIGH"
        # Check RUL score
        for line in rca_text.split("\n"):
            if "rul score" in line.lower():
                try:
                    score = float(line.split(":")[-1].strip().split()[0])
                    if score <= 0.10:
                        return "CRITICAL"
                    elif score <= 0.25:
                        return "HIGH"
                    elif score <= 0.50:
                        return "MEDIUM"
                    else:
                        return "LOW"
                except (ValueError, IndexError):
                    pass
        return "MEDIUM"

    def _get_recommended_actions(self, fault_type: str,
                                  urgency: str,
                                  cmms_summary: dict) -> list[str]:
        """Generate recommended actions based on fault type and context."""
        actions = []

        # Universal first action based on urgency
        if urgency == "CRITICAL":
            actions.append(
                "IMMEDIATE: Initiate controlled shutdown of the equipment. "
                "Do not continue operation — risk of catastrophic failure "
                "and secondary equipment damage."
            )
        elif urgency == "HIGH":
            actions.append(
                "Schedule emergency maintenance within 48 hours. "
                "Reduce operational load if possible until maintenance "
                "is performed."
            )

        # Fault-specific actions
        if "inner race" in fault_type:
            actions.extend([
                "Inspect lubrication condition — inadequate grease is "
                "the primary cause of inner race spalling (60% of cases). "
                "Check re-greasing log in CMMS.",
                "Verify shaft alignment and interference fit — excessive "
                "preload accelerates inner race fatigue.",
                "Order replacement bearing and schedule installation "
                "during next available maintenance window.",
            ])
        elif "outer race" in fault_type:
            actions.extend([
                "Verify bearing housing bore tolerance — race distortion "
                "from improper fit is the leading cause of outer race defects.",
                "Inspect seals for moisture ingress — corrosion from "
                "humidity causes outer race pitting.",
                "Check bearing temperature trend — if rising, schedule "
                "immediate replacement.",
            ])
        elif "ball" in fault_type or "rolling element" in fault_type:
            actions.extend([
                "Flush and replace lubricant — particle contamination "
                "(>5 microns) initiates surface pitting on rolling elements.",
                "Inspect for electrical pitting marks (frosting pattern) — "
                "common in VFD-driven motors from stray current discharge.",
                "Verify load calculations against bearing rating — "
                "material fatigue from overloading causes BSF defects.",
            ])
        elif "impulsive" in fault_type or "spalling" in fault_type:
            actions.extend([
                "Perform detailed vibration spectrum analysis to confirm "
                "specific fault frequency (BPFI/BPFO/BSF).",
                "Inspect bearing for visible spalling — if confirmed, "
                "schedule replacement before next planned shutdown.",
                "Review lubrication schedule — adjust re-grease interval "
                "if currently exceeding manufacturer recommendation.",
            ])
        else:
            actions.extend([
                "Perform detailed vibration spectrum analysis.",
                "Inspect bearing and lubrication condition.",
                "Review maintenance history for recurring patterns.",
            ])

        # Parts check
        parts = cmms_summary.get("spare_parts", [])
        bearing_parts = [p for p in parts
                         if p.get("category") == "bearing" and
                         "error" not in p]
        if bearing_parts:
            in_stock = [p for p in bearing_parts if p.get("in_stock")]
            if in_stock:
                p = in_stock[0]
                actions.append(
                    f"Verify replacement bearing {p['part_number']} "
                    f"({p['qty_available']} in stock at "
                    f"{p['location']}). Reserve one unit for this work order."
                )
            else:
                actions.append(
                    "ORDER IMMEDIATELY: Replacement bearing is OUT OF STOCK. "
                    "Initiate emergency procurement. "
                    f"Lead time: {bearing_parts[0].get('lead_time_days', '?')} "
                    "days."
                )

        # Documentation action
        actions.append(
            "Document all findings and corrective actions in CMMS work order. "
            "Record baseline vibration after maintenance for future comparison."
        )

        return actions


# ── Convenience: run full pipeline ────────────────────────────────────────────

def run_rca_pipeline(feature_matrix_path: str,
                     models_dir: str,
                     snapshot_index: int = None,
                     bearing_id: str = None,
                     output_dir: str = "results/rca",
                     api_key: str = None) -> dict:
    """
    Run the complete RCA pipeline end-to-end.

    This is the function you call to go from raw data to RCA report:
        feature_matrix.csv → SHAP → MCP queries → RCA report

    Args:
        feature_matrix_path : path to feature_matrix.csv
        models_dir          : directory with IF + LSTM model files
        snapshot_index      : which snapshot to analyze (default: last)
        bearing_id          : specific bearing (default: most anomalous)
        output_dir          : where to save the report
        api_key             : Anthropic API key (optional)

    Returns:
        RCA report dict
    """
    import pandas as pd

    # These imports are here to avoid circular imports at module level
    # and to keep the module runnable standalone
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from isolation_forest import BearingAnomalyDetector, SingleBearingDetector
    from rul_lstm import BearingRULPredictor, SingleBearingRUL, LSTMRULModel
    from shap_explainer import BearingShapExplainer
    from mcp_equipment_manual import EquipmentManualMCP
    from mcp_cmms import CMMSMCP
    import __main__
    for cls in [BearingAnomalyDetector, SingleBearingDetector,
                BearingRULPredictor, SingleBearingRUL, LSTMRULModel]:
        setattr(__main__, cls.__name__, cls)
    print("=" * 60)
    print("BearingMind — Full RCA Pipeline")
    print("=" * 60)

    # ── Load feature matrix ───────────────────────────────────────────
    print("\n[1/6] Loading feature matrix ...")
    df = pd.read_csv(feature_matrix_path, index_col=0)
    if snapshot_index is None:
        snapshot_index = len(df) - 1
    print(f"  {df.shape[0]} snapshots × {df.shape[1]} features")
    print(f"  Analyzing snapshot: {snapshot_index}")

    # ── Load ML models ────────────────────────────────────────────────
    print("\n[2/6] Loading ML models ...")
    if_dir  = os.path.join(models_dir, "if", "models")
    rul_dir = os.path.join(models_dir, "rul", "models")

    anomaly_det = BearingAnomalyDetector()
    anomaly_det.fit_from_df(df)
    if os.path.isdir(if_dir):
        anomaly_det.load_models(if_dir)
        print(f"  IF models loaded from {if_dir}")

    rul_pred = BearingRULPredictor()
    if os.path.isdir(rul_dir):
        rul_pred.feature_matrix_ = df
        rul_pred.load_models(rul_dir)
        rul_pred.bearing_ids_ = sorted(rul_pred.predictors_.keys())
        print(f"  LSTM models loaded from {rul_dir}")

    # ── Run SHAP ──────────────────────────────────────────────────────
    print("\n[3/6] Running SHAP explainer ...")
    explainer = BearingShapExplainer(anomaly_det, rul_pred)
    explainer.fit(df=df, n_background=100)
    shap_result = explainer.explain_snapshot(snapshot_index)
    print(f"  Most anomalous bearing: "
          f"{shap_result['most_anomalous_bearing']}")

    # ── Initialize MCP servers ────────────────────────────────────────
    print("\n[4/6] Initializing MCP servers ...")
    manual_mcp = EquipmentManualMCP()
    manual_mcp.load()

    cmms_db = os.path.join(output_dir, "cmms.db")
    cmms_mcp = CMMSMCP(db_path=cmms_db)
    cmms_mcp.initialize()

    # ── Run RCA Agent ─────────────────────────────────────────────────
    print("\n[5/6] Running RCA Agent ...")
    rca = RCAAgent(
        manual_mcp=manual_mcp,
        cmms_mcp=cmms_mcp,
        api_key=api_key,
    )

    # Use specific bearing or most anomalous
    target_bearing = bearing_id or shap_result["most_anomalous_bearing"]
    target_context = shap_result.get(target_bearing, shap_result)

    report = rca.analyze(target_context, bearing_id=target_bearing)

    # ── Save report ───────────────────────────────────────────────────
    print("\n[6/6] Saving report ...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Text report
    report_path = os.path.join(output_dir, "rca_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report["report_text"])
    print(f"  Report saved → {report_path}")

    # JSON metadata
    meta_path = os.path.join(output_dir, "rca_metadata.json")
    meta = {
        "bearing_id":      report["bearing_id"],
        "fault_type":      report["fault_type"],
        "urgency":         report["urgency"],
        "generation_mode": report["generation_mode"],
        "snapshot_index":  snapshot_index,
        "mcp_calls_made":  report["mcp_calls_made"],
        "recommended_actions": report["recommended_actions"],
        "timestamp":       datetime.now().isoformat(),
    }
    with open(meta_path, "w", encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {meta_path}")

    cmms_mcp.close()

    print(f"\n{'='*60}")
    print("✓ RCA Pipeline complete")
    print(f"  Urgency: {report['urgency']}")
    print(f"  Mode: {report['generation_mode']}")
    print(f"{'='*60}")

    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from isolation_forest import BearingAnomalyDetector, SingleBearingDetector
    from rul_lstm import BearingRULPredictor, SingleBearingRUL, LSTMRULModel    

    if len(sys.argv) < 3:
        print("Usage: python rca_agent.py <feature_matrix.csv> "
              "<models_dir> [snapshot_index] [output_dir]")
        print("  feature_matrix.csv : output from features.py")
        print("  models_dir         : directory with IF + LSTM models")
        print("  snapshot_index     : snapshot to analyze (default: last)")
        print("  output_dir         : where to save report "
              "(default: results/rca)")
        print("\nSet ANTHROPIC_API_KEY env var for LLM-generated reports.")
        sys.exit(1)

    feature_csv = sys.argv[1]
    models_dir  = sys.argv[2]
    snap_idx    = int(sys.argv[3]) if len(sys.argv) > 3 else None
    out_dir     = sys.argv[4] if len(sys.argv) > 4 else "results/rca"

    report = run_rca_pipeline(
        feature_matrix_path=feature_csv,
        models_dir=models_dir,
        snapshot_index=snap_idx,
        output_dir=out_dir,
    )

    print("\n" + report["report_text"])
