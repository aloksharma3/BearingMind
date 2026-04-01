"""
alert_agent.py — Alert & Notification Agent
Industrial AI Predictive Maintenance | BearingMind

Takes an RCA report and routes it: who gets notified, how urgently,
through what channel. Auto-creates CMMS work orders for HIGH/CRITICAL.
Maintains an alert log for the dashboard and audit trail.

Routing rules:
    CRITICAL → SMS + email (supervisor, plant manager) + Slack #critical-alerts
               + auto-create EMERGENCY work order
    HIGH     → Email (planner, reliability eng) + Slack #maintenance
               + auto-create HIGH priority work order
    MEDIUM   → Email (planner) + dashboard log
    LOW      → Dashboard log only

Usage:
    from src.alert_agent import AlertAgent

    alert = AlertAgent(cmms_mcp=cmms_mcp)
    result = alert.process(rca_report)
    print(result["summary"])
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Routing configuration ─────────────────────────────────────────────────────

CONTACTS = {
    "supervisor":  {"name": "R. Martinez", "role": "Maintenance Supervisor",
                    "email": "r.martinez@plant.example.com",
                    "phone": "+1-513-555-0142"},
    "planner":     {"name": "J. Chen", "role": "Maintenance Planner",
                    "email": "j.chen@plant.example.com",
                    "phone": "+1-513-555-0198"},
    "plant_mgr":   {"name": "S. Williams", "role": "Plant Manager",
                    "email": "s.williams@plant.example.com",
                    "phone": "+1-513-555-0101"},
    "reliability":  {"name": "A. Kumar", "role": "Reliability Engineer",
                    "email": "a.kumar@plant.example.com",
                    "phone": "+1-513-555-0176"},
}

ROUTING = {
    "CRITICAL": {
        "notify": ["supervisor", "plant_mgr", "reliability"],
        "channels": ["sms", "email", "slack:#critical-alerts"],
        "create_wo": "EMERGENCY",
        "response": "Immediate — controlled shutdown required",
    },
    "HIGH": {
        "notify": ["planner", "reliability"],
        "channels": ["email", "slack:#maintenance"],
        "create_wo": "HIGH",
        "response": "Within 48 hours",
    },
    "MEDIUM": {
        "notify": ["planner"],
        "channels": ["email"],
        "create_wo": None,
        "response": "Next planned maintenance window",
    },
    "LOW": {
        "notify": [],
        "channels": [],
        "create_wo": None,
        "response": "Monitor and trend — no action required",
    },
}


# ── Alert Agent ───────────────────────────────────────────────────────────────

class AlertAgent:
    """
    Routes RCA reports to the right people based on urgency.

    Args:
        cmms_mcp  : initialized CMMSMCP instance (optional — for WO creation)
        log_path  : path to alert log JSON file
    """

    BEARING_TO_ASSET = {
        "b1_ch1": "BRG-001", "b2_ch1": "BRG-002",
        "b3_ch1": "BRG-003", "b4_ch1": "BRG-004",
    }

    def __init__(self, cmms_mcp=None,
                 log_path: str = "results/rca/alert_log.json"):
        self.cmms_mcp = cmms_mcp
        self.log_path = log_path

    def process(self, rca_report: dict) -> dict:
        """
        Route an RCA report — notify, create work order, log.

        Args:
            rca_report : output from RCAAgent.analyze()

        Returns:
            dict with: summary, notifications, work_order, log_entry
        """
        urgency    = rca_report.get("urgency", "MEDIUM")
        bearing_id = rca_report.get("bearing_id", "unknown")
        fault_type = rca_report.get("fault_type", "undetermined")
        actions    = rca_report.get("recommended_actions", [])
        routing    = ROUTING.get(urgency, ROUTING["MEDIUM"])

        print(f"\nAlert Agent — {urgency} alert for {bearing_id}")

        # ── Build notifications ───────────────────────────────────────
        notifications = []
        for contact_key in routing["notify"]:
            contact = CONTACTS.get(contact_key, {})
            notif = {
                "recipient":  contact.get("name", contact_key),
                "role":       contact.get("role", ""),
                "email":      contact.get("email", ""),
                "channels":   routing["channels"],
                "subject":    f"[{urgency}] {fault_type} on {bearing_id}",
                "message": (
                    f"{fault_type.upper()} detected on {bearing_id}. "
                    f"{routing['response']}. "
                    f"Top action: {actions[0] if actions else 'See report.'}"
                ),
                "timestamp":  datetime.now().isoformat(),
            }
            notifications.append(notif)
            print(f"  → {contact.get('name', contact_key)} "
                  f"via {', '.join(routing['channels'])}")

        if not notifications:
            print("  → Log only (no notifications)")

        # ── Create work order if required ─────────────────────────────
        work_order = None
        if routing["create_wo"]:
            work_order = self._create_work_order(
                bearing_id, fault_type, urgency,
                routing["create_wo"], actions)
            print(f"  → Work order {work_order['wo_number']} "
                  f"({routing['create_wo']})")

        # ── Log ───────────────────────────────────────────────────────
        log_entry = {
            "timestamp":    datetime.now().isoformat(),
            "bearing_id":   bearing_id,
            "fault_type":   fault_type,
            "urgency":      urgency,
            "response":     routing["response"],
            "notified":     [n["recipient"] for n in notifications],
            "channels":     routing["channels"],
            "work_order":   work_order["wo_number"] if work_order else None,
        }
        self._append_log(log_entry)

        # ── Summary ───────────────────────────────────────────────────
        wo_text = f", WO {work_order['wo_number']}" if work_order else ""
        summary = (
            f"[{urgency}] {fault_type.upper()} on {bearing_id} — "
            f"{len(notifications)} notified{wo_text} — "
            f"{routing['response']}"
        )

        return {
            "summary":       summary,
            "urgency":       urgency,
            "notifications": notifications,
            "work_order":    work_order,
            "log_entry":     log_entry,
        }

    def _create_work_order(self, bearing_id: str, fault_type: str,
                            urgency: str, priority: str,
                            actions: list) -> dict:
        """Create a work order in CMMS (if connected) or return WO dict."""
        now = datetime.now()
        wo = {
            "wo_number":    f"WO-{now.strftime('%Y-%m%d')}-AI",
            "asset_id":     self.BEARING_TO_ASSET.get(bearing_id, bearing_id),
            "wo_type":      "CM",
            "priority":     priority,
            "created_date": now.strftime("%Y-%m-%d"),
            "created_by":   "BearingMind AI",
            "description":  f"AI-generated: {fault_type} on {bearing_id} "
                            f"(urgency: {urgency})",
            "actions":      actions[:5],
        }

        if self.cmms_mcp and self.cmms_mcp.is_initialized_:
            try:
                self.cmms_mcp.conn_.execute("""
                    INSERT OR IGNORE INTO work_orders
                    (wo_number, asset_id, wo_type, priority,
                     created_date, completed_date, description,
                     findings, actions_taken, technician, labor_hours)
                    VALUES (?, ?, ?, ?, ?, NULL, ?, ?, '', 'BearingMind AI', 0)
                """, (wo["wo_number"], wo["asset_id"], wo["wo_type"],
                      wo["priority"], wo["created_date"],
                      wo["description"],
                      f"Auto-detected {fault_type} via SHAP analysis."))
                self.cmms_mcp.conn_.commit()
                wo["cmms_inserted"] = True
            except Exception as e:
                wo["cmms_inserted"] = False
        else:
            wo["cmms_inserted"] = False

        return wo

    def _append_log(self, entry: dict) -> None:
        """Append to alert log JSON file."""
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        log = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    log = json.load(f)
            except (json.JSONDecodeError, IOError):
                log = []
        log.append(entry)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

    def get_history(self, n: int = 10) -> list[dict]:
        """Return last N alert log entries."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            return json.load(f)[-n:]


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Alert Agent — BearingMind")
    print("=" * 60)

    test_reports = [
        {"urgency": "CRITICAL", "bearing_id": "b1_ch1",
         "fault_type": "outer race fault",
         "recommended_actions": [
             "IMMEDIATE: Initiate controlled shutdown.",
             "Verify housing bore tolerance.",
             "Replacement bearing RX-ZA2115 in stock (3 available)."]},
        {"urgency": "HIGH", "bearing_id": "b2_ch1",
         "fault_type": "inner race fault",
         "recommended_actions": [
             "Schedule maintenance within 48 hours.",
             "Inspect lubrication condition."]},
        {"urgency": "LOW", "bearing_id": "b3_ch1",
         "fault_type": "undetermined",
         "recommended_actions": ["Monitor and trend."]},
    ]

    alert = AlertAgent(log_path="results/rca/alert_log.json")

    for report in test_reports:
        result = alert.process(report)
        print(f"  Summary: {result['summary']}\n")

    print(f"Alert history: {len(alert.get_history())} entries logged")
