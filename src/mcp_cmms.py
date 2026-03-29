"""
mcp_cmms.py — CMMS (Computerized Maintenance Management System) MCP Server
Industrial AI Predictive Maintenance | BearingMind

An MCP server that simulates a plant CMMS database (like SAP PM, Maximo,
or Fiix) using SQLite. Exposes maintenance history, asset registry, and
spare parts inventory as queryable tools for the RCA agent.

What this does:
    The RCA agent needs operational context that raw sensor data can't
    provide. When it detects an outer race fault on Bearing 1, it asks:
        - "When was the last work order on this asset?"
        - "What maintenance was performed?"
        - "Is the replacement bearing in stock?"
    This server answers those questions from a realistic mock database.

Database schema:
    assets        — equipment registry (bearing ID, model, install date)
    work_orders   — maintenance history (type, date, description, findings)
    spare_parts   — parts inventory (SKU, quantity, location, reorder level)

MCP tools exposed:
    get_asset_info(asset_id)        → asset details + install date
    get_work_orders(asset_id, n)    → last N work orders for an asset
    check_spare_parts(part_number)  → stock status for a part
    get_maintenance_summary(asset_id) → combined asset + recent WO + parts

Why SQLite:
    - Real CMMS systems use SQL databases (Oracle, SQL Server, PostgreSQL)
    - SQLite gives us the same SQL interface with zero infrastructure
    - The MCP tool interface is identical whether backed by SQLite or SAP
    - Hiring managers see "SQL-backed CMMS integration" — not "JSON mock"

Pipeline position:
    SHAP explainer → RCA Agent → [get_maintenance_summary] → CMMS MCP
                                                               ↓
                                                    work order history,
                                                    parts availability
                                                               ↓
                                                    RCA Agent includes
                                                    in fault report

Usage:
    from src.mcp_cmms import CMMSMCP

    cmms = CMMSMCP(db_path="data/cmms.db")
    cmms.initialize()   # creates tables + loads mock data

    # RCA agent calls
    info    = cmms.get_asset_info("BRG-001")
    history = cmms.get_work_orders("BRG-001", n=5)
    stock   = cmms.check_spare_parts("SKF-6205-2RS")
    summary = cmms.get_maintenance_summary("BRG-001")
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ── Mock data generator ───────────────────────────────────────────────────────

def _generate_mock_data() -> dict:
    """
    Generate realistic CMMS mock data matching the NASA IMS test rig.

    The NASA IMS 2nd test dataset has 4 bearings on a single shaft:
        - Bearing 1 (b1_ch1) — Channel 1
        - Bearing 2 (b2_ch1) — Channel 2
        - Bearing 3 (b3_ch1) — Channel 3
        - Bearing 4 (b4_ch1) — Channel 4

    Test ran from 2004-02-12 to 2004-02-19 (about 7 days).
    Bearing 1 had an outer race defect at end of test.

    We create a realistic maintenance history going back 2 years
    before the test, as a real plant would have.
    """

    # ── Assets ────────────────────────────────────────────────────────────
    assets = [
        {
            "asset_id":      "BRG-001",
            "bearing_id":    "b1_ch1",
            "description":   "Drive-end bearing, Test Rig Shaft 1",
            "model":         "Rexnord ZA-2115 double row bearing",
            "manufacturer":  "Rexnord",
            "install_date":  "2003-06-15",
            "location":      "Test Rig — Position 1 (drive end)",
            "criticality":   "A",
            "rated_speed":   2000,  # RPM
            "rated_load_kn": 26.7,  # kN radial load
            "status":        "RUNNING",
        },
        {
            "asset_id":      "BRG-002",
            "bearing_id":    "b2_ch1",
            "description":   "Inner bearing 1, Test Rig Shaft 1",
            "model":         "Rexnord ZA-2115 double row bearing",
            "manufacturer":  "Rexnord",
            "install_date":  "2003-06-15",
            "location":      "Test Rig — Position 2 (inner 1)",
            "criticality":   "A",
            "rated_speed":   2000,
            "rated_load_kn": 26.7,
            "status":        "RUNNING",
        },
        {
            "asset_id":      "BRG-003",
            "bearing_id":    "b3_ch1",
            "description":   "Inner bearing 2, Test Rig Shaft 1",
            "model":         "Rexnord ZA-2115 double row bearing",
            "manufacturer":  "Rexnord",
            "install_date":  "2003-06-15",
            "location":      "Test Rig — Position 3 (inner 2)",
            "criticality":   "A",
            "rated_speed":   2000,
            "rated_load_kn": 26.7,
            "status":        "RUNNING",
        },
        {
            "asset_id":      "BRG-004",
            "bearing_id":    "b4_ch1",
            "description":   "Free-end bearing, Test Rig Shaft 1",
            "model":         "Rexnord ZA-2115 double row bearing",
            "manufacturer":  "Rexnord",
            "install_date":  "2003-06-15",
            "location":      "Test Rig — Position 4 (free end)",
            "criticality":   "A",
            "rated_speed":   2000,
            "rated_load_kn": 26.7,
            "status":        "RUNNING",
        },
    ]

    # ── Work Orders ───────────────────────────────────────────────────────
    # Realistic maintenance history going back before the test
    work_orders = [
        # BRG-001 history
        {
            "wo_number":    "WO-2003-0142",
            "asset_id":     "BRG-001",
            "wo_type":      "PM",
            "priority":     "ROUTINE",
            "created_date": "2003-07-15",
            "completed_date": "2003-07-15",
            "description":  "Initial commissioning inspection — all bearings",
            "findings":     "All bearings installed correctly. Shaft alignment "
                            "within spec (0.03mm offset, 0.02mm angular). "
                            "Initial vibration baseline recorded: RMS 0.42 mm/s.",
            "actions_taken": "Recorded baseline vibration. Verified grease fill "
                             "level at 40% cavity volume. Logged initial "
                             "temperature: 34°C at 2000 RPM steady state.",
            "technician":   "R. Martinez",
            "labor_hours":  2.5,
        },
        {
            "wo_number":    "WO-2003-0298",
            "asset_id":     "BRG-001",
            "wo_type":      "PM",
            "priority":     "ROUTINE",
            "created_date": "2003-10-20",
            "completed_date": "2003-10-20",
            "description":  "Quarterly vibration route — all test rig bearings",
            "findings":     "BRG-001 vibration stable at RMS 0.45 mm/s. "
                            "No significant change from baseline. Bearing "
                            "temperature 36°C. Grease condition: acceptable.",
            "actions_taken": "Re-greased with Mobilith SHC 220 (15ml). "
                             "Updated vibration trend in monitoring system.",
            "technician":   "R. Martinez",
            "labor_hours":  1.5,
        },
        {
            "wo_number":    "WO-2004-0015",
            "asset_id":     "BRG-001",
            "wo_type":      "PM",
            "priority":     "ROUTINE",
            "created_date": "2004-01-18",
            "completed_date": "2004-01-18",
            "description":  "Quarterly vibration route + grease sample analysis",
            "findings":     "BRG-001 RMS 0.51 mm/s — slight increase from "
                            "baseline but within normal band. Grease sample "
                            "sent to lab: no metallic particles detected. "
                            "Bearing temperature 37°C.",
            "actions_taken": "Re-greased (15ml). Lab results pending. "
                             "Noted minor surface rust on housing exterior — "
                             "recommend environmental seal inspection next PM.",
            "technician":   "J. Chen",
            "labor_hours":  2.0,
        },

        # BRG-002 history
        {
            "wo_number":    "WO-2003-0143",
            "asset_id":     "BRG-002",
            "wo_type":      "PM",
            "priority":     "ROUTINE",
            "created_date": "2003-07-15",
            "completed_date": "2003-07-15",
            "description":  "Initial commissioning inspection",
            "findings":     "Bearing installed correctly. Baseline vibration "
                            "RMS 0.38 mm/s. Temperature 33°C.",
            "actions_taken": "Recorded baseline. Verified grease fill.",
            "technician":   "R. Martinez",
            "labor_hours":  1.0,
        },
        {
            "wo_number":    "WO-2003-0299",
            "asset_id":     "BRG-002",
            "wo_type":      "PM",
            "priority":     "ROUTINE",
            "created_date": "2003-10-20",
            "completed_date": "2003-10-20",
            "description":  "Quarterly vibration route",
            "findings":     "BRG-002 stable. RMS 0.40 mm/s. No concerns.",
            "actions_taken": "Re-greased (15ml).",
            "technician":   "R. Martinez",
            "labor_hours":  1.0,
        },

        # BRG-003 history
        {
            "wo_number":    "WO-2003-0144",
            "asset_id":     "BRG-003",
            "wo_type":      "PM",
            "priority":     "ROUTINE",
            "created_date": "2003-07-15",
            "completed_date": "2003-07-15",
            "description":  "Initial commissioning inspection",
            "findings":     "Bearing installed correctly. Baseline RMS 0.41 mm/s.",
            "actions_taken": "Recorded baseline. Verified grease fill.",
            "technician":   "R. Martinez",
            "labor_hours":  1.0,
        },

        # BRG-004 history
        {
            "wo_number":    "WO-2003-0145",
            "asset_id":     "BRG-004",
            "wo_type":      "PM",
            "priority":     "ROUTINE",
            "created_date": "2003-07-15",
            "completed_date": "2003-07-15",
            "description":  "Initial commissioning inspection",
            "findings":     "Bearing installed correctly. Baseline RMS 0.44 mm/s.",
            "actions_taken": "Recorded baseline. Verified grease fill.",
            "technician":   "R. Martinez",
            "labor_hours":  1.0,
        },
        {
            "wo_number":    "WO-2004-0022",
            "asset_id":     "BRG-004",
            "wo_type":      "CM",
            "priority":     "HIGH",
            "created_date": "2004-01-25",
            "completed_date": "2004-01-26",
            "description":  "Corrective maintenance — elevated temperature alarm",
            "findings":     "BRG-004 temperature reached 52°C during high-load "
                            "test. Vibration RMS 0.78 mm/s. Bearing inspection "
                            "showed minor discoloration on outer race — "
                            "consistent with thermal stress.",
            "actions_taken": "Cleaned and re-lubricated with fresh grease. "
                             "Reduced test load by 10% until next scheduled "
                             "inspection. Added to watchlist for increased "
                             "monitoring frequency (weekly instead of quarterly).",
            "technician":   "J. Chen",
            "labor_hours":  3.5,
        },
    ]

    # ── Spare Parts ───────────────────────────────────────────────────────
    spare_parts = [
        {
            "part_number":   "RX-ZA2115",
            "description":   "Rexnord ZA-2115 double row bearing",
            "category":      "bearing",
            "unit_cost":     285.00,
            "qty_on_hand":   3,
            "qty_reserved":  0,
            "reorder_level": 2,
            "reorder_qty":   4,
            "location":      "Warehouse A — Shelf B3-07",
            "lead_time_days": 5,
            "compatible_assets": "BRG-001,BRG-002,BRG-003,BRG-004",
            "last_received": "2003-11-10",
        },
        {
            "part_number":   "SKF-6205-2RS",
            "description":   "SKF 6205-2RS sealed deep groove ball bearing",
            "category":      "bearing",
            "unit_cost":     42.50,
            "qty_on_hand":   8,
            "qty_reserved":  1,
            "reorder_level": 4,
            "reorder_qty":   10,
            "location":      "Warehouse A — Shelf B3-02",
            "lead_time_days": 3,
            "compatible_assets": "MOTOR-001,PUMP-002",
            "last_received": "2004-01-05",
        },
        {
            "part_number":   "MOBIL-SHC220-1KG",
            "description":   "Mobilith SHC 220 lithium complex grease (1kg)",
            "category":      "lubricant",
            "unit_cost":     38.75,
            "qty_on_hand":   5,
            "qty_reserved":  0,
            "reorder_level": 3,
            "reorder_qty":   6,
            "location":      "Warehouse A — Lubricant Cabinet LC-01",
            "lead_time_days": 2,
            "compatible_assets": "BRG-001,BRG-002,BRG-003,BRG-004",
            "last_received": "2003-12-20",
        },
        {
            "part_number":   "SEAL-ZA2115-V",
            "description":   "Viton seal ring for ZA-2115 bearing housing",
            "category":      "seal",
            "unit_cost":     12.00,
            "qty_on_hand":   6,
            "qty_reserved":  0,
            "reorder_level": 4,
            "reorder_qty":   8,
            "location":      "Warehouse A — Shelf B3-08",
            "lead_time_days": 7,
            "compatible_assets": "BRG-001,BRG-002,BRG-003,BRG-004",
            "last_received": "2003-09-01",
        },
        {
            "part_number":   "PCB-353B33",
            "description":   "PCB Piezotronics 353B33 accelerometer",
            "category":      "sensor",
            "unit_cost":     695.00,
            "qty_on_hand":   2,
            "qty_reserved":  0,
            "reorder_level": 1,
            "reorder_qty":   2,
            "location":      "Instrument Lab — Cabinet IC-03",
            "lead_time_days": 14,
            "compatible_assets": "BRG-001,BRG-002,BRG-003,BRG-004",
            "last_received": "2003-06-01",
        },
    ]

    return {
        "assets": assets,
        "work_orders": work_orders,
        "spare_parts": spare_parts,
    }


# ── Bearing ID ↔ Asset ID mapping ────────────────────────────────────────────

BEARING_TO_ASSET = {
    "b1_ch1": "BRG-001",
    "b2_ch1": "BRG-002",
    "b3_ch1": "BRG-003",
    "b4_ch1": "BRG-004",
}

ASSET_TO_BEARING = {v: k for k, v in BEARING_TO_ASSET.items()}


# ── CMMS MCP Server ──────────────────────────────────────────────────────────

class CMMSMCP:
    """
    MCP server backed by a SQLite CMMS database.

    Provides maintenance context to the RCA agent:
        - Asset information (what equipment, when installed)
        - Work order history (what maintenance has been done)
        - Spare parts availability (can we replace it now?)

    Args:
        db_path : path to SQLite database file
                  (created with mock data if it doesn't exist)
    """

    # ── MCP tool schemas ──────────────────────────────────────────────────
    TOOL_SCHEMAS = [
        {
            "name": "get_asset_info",
            "description": (
                "Get equipment details for a bearing asset including "
                "model, install date, location, and criticality rating."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": (
                            "Asset ID (e.g. 'BRG-001') or bearing ID "
                            "(e.g. 'b1_ch1')"
                        ),
                    },
                },
                "required": ["asset_id"],
            },
        },
        {
            "name": "get_work_orders",
            "description": (
                "Get maintenance work order history for an asset. "
                "Returns the most recent N work orders with dates, "
                "findings, and actions taken."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Asset ID or bearing ID",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of recent work orders",
                        "default": 5,
                    },
                },
                "required": ["asset_id"],
            },
        },
        {
            "name": "check_spare_parts",
            "description": (
                "Check spare parts inventory for a given part number "
                "or search for parts compatible with an asset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_number": {
                        "type": "string",
                        "description": (
                            "Part number (e.g. 'RX-ZA2115') or asset ID "
                            "to find all compatible parts"
                        ),
                    },
                },
                "required": ["part_number"],
            },
        },
        {
            "name": "get_maintenance_summary",
            "description": (
                "Get a complete maintenance summary for an asset: "
                "asset details, recent work orders, and spare parts "
                "availability. This is the primary tool for the RCA agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Asset ID or bearing ID",
                    },
                },
                "required": ["asset_id"],
            },
        },
    ]

    def __init__(self, db_path: str = "data/cmms.db"):
        self.db_path = db_path
        self.conn_: sqlite3.Connection | None = None
        self.is_initialized_ = False

    def _resolve_asset_id(self, id_str: str) -> str:
        """Convert bearing_id (b1_ch1) to asset_id (BRG-001) if needed."""
        if id_str in BEARING_TO_ASSET:
            return BEARING_TO_ASSET[id_str]
        return id_str

    # ── Initialization ────────────────────────────────────────────────────

    def initialize(self, force_rebuild: bool = False) -> "CMMSMCP":
        """
        Create SQLite database with schema and load mock data.

        If the database already exists and force_rebuild is False,
        just connects to it. Otherwise creates fresh tables.
        """
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        db_exists = os.path.exists(self.db_path)
        self.conn_ = sqlite3.connect(self.db_path)
        self.conn_.row_factory = sqlite3.Row  # dict-like row access

        if not db_exists or force_rebuild:
            self._create_schema()
            self._load_mock_data()
            print(f"  CMMS database created → {self.db_path}")
        else:
            print(f"  CMMS database loaded → {self.db_path}")

        self.is_initialized_ = True
        return self

    def _create_schema(self):
        """Create CMMS database tables."""
        cur = self.conn_.cursor()

        cur.executescript("""
            DROP TABLE IF EXISTS assets;
            DROP TABLE IF EXISTS work_orders;
            DROP TABLE IF EXISTS spare_parts;

            CREATE TABLE assets (
                asset_id        TEXT PRIMARY KEY,
                bearing_id      TEXT,
                description     TEXT,
                model           TEXT,
                manufacturer    TEXT,
                install_date    TEXT,
                location        TEXT,
                criticality     TEXT,
                rated_speed     INTEGER,
                rated_load_kn   REAL,
                status          TEXT
            );

            CREATE TABLE work_orders (
                wo_number       TEXT PRIMARY KEY,
                asset_id        TEXT,
                wo_type         TEXT,     -- PM / CM / EM (preventive/corrective/emergency)
                priority        TEXT,     -- ROUTINE / HIGH / CRITICAL
                created_date    TEXT,
                completed_date  TEXT,
                description     TEXT,
                findings        TEXT,
                actions_taken   TEXT,
                technician      TEXT,
                labor_hours     REAL,
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
            );

            CREATE TABLE spare_parts (
                part_number     TEXT PRIMARY KEY,
                description     TEXT,
                category        TEXT,
                unit_cost       REAL,
                qty_on_hand     INTEGER,
                qty_reserved    INTEGER,
                reorder_level   INTEGER,
                reorder_qty     INTEGER,
                location        TEXT,
                lead_time_days  INTEGER,
                compatible_assets TEXT,   -- comma-separated asset IDs
                last_received   TEXT
            );

            CREATE INDEX idx_wo_asset ON work_orders(asset_id);
            CREATE INDEX idx_wo_date  ON work_orders(completed_date);
            CREATE INDEX idx_parts_cat ON spare_parts(category);
        """)
        self.conn_.commit()

    def _load_mock_data(self):
        """Insert mock data into tables."""
        data = _generate_mock_data()
        cur = self.conn_.cursor()

        for asset in data["assets"]:
            cur.execute("""
                INSERT INTO assets VALUES
                (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                asset["asset_id"], asset["bearing_id"],
                asset["description"], asset["model"],
                asset["manufacturer"], asset["install_date"],
                asset["location"], asset["criticality"],
                asset["rated_speed"], asset["rated_load_kn"],
                asset["status"],
            ))

        for wo in data["work_orders"]:
            cur.execute("""
                INSERT INTO work_orders VALUES
                (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                wo["wo_number"], wo["asset_id"], wo["wo_type"],
                wo["priority"], wo["created_date"],
                wo["completed_date"], wo["description"],
                wo["findings"], wo["actions_taken"],
                wo["technician"], wo["labor_hours"],
            ))

        for part in data["spare_parts"]:
            cur.execute("""
                INSERT INTO spare_parts VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                part["part_number"], part["description"],
                part["category"], part["unit_cost"],
                part["qty_on_hand"], part["qty_reserved"],
                part["reorder_level"], part["reorder_qty"],
                part["location"], part["lead_time_days"],
                part["compatible_assets"], part["last_received"],
            ))

        self.conn_.commit()
        print(f"    {len(data['assets'])} assets, "
              f"{len(data['work_orders'])} work orders, "
              f"{len(data['spare_parts'])} spare parts loaded")

    # ── MCP Tool implementations ──────────────────────────────────────────

    def get_asset_info(self, asset_id: str) -> dict:
        """
        Get asset details.

        Returns dict with all asset fields or error message.
        """
        if not self.is_initialized_:
            raise RuntimeError("Call initialize() first.")

        asset_id = self._resolve_asset_id(asset_id)
        cur = self.conn_.cursor()
        cur.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,))
        row = cur.fetchone()

        if row is None:
            return {"error": f"Asset '{asset_id}' not found in CMMS."}

        result = dict(row)
        # Add computed field: days since installation
        install = datetime.strptime(result["install_date"], "%Y-%m-%d")
        result["days_in_service"] = (datetime(2004, 2, 15) - install).days
        return result

    def get_work_orders(self, asset_id: str, n: int = 5) -> list[dict]:
        """
        Get the N most recent work orders for an asset.

        Returns list of work order dicts, newest first.
        """
        if not self.is_initialized_:
            raise RuntimeError("Call initialize() first.")

        asset_id = self._resolve_asset_id(asset_id)
        cur = self.conn_.cursor()
        cur.execute("""
            SELECT * FROM work_orders
            WHERE asset_id = ?
            ORDER BY completed_date DESC
            LIMIT ?
        """, (asset_id, n))

        return [dict(row) for row in cur.fetchall()]

    def check_spare_parts(self, part_number: str) -> list[dict]:
        """
        Check spare parts inventory.

        If part_number matches a part number → return that part.
        If part_number matches an asset ID → return all compatible parts.
        """
        if not self.is_initialized_:
            raise RuntimeError("Call initialize() first.")

        part_number = self._resolve_asset_id(part_number)
        cur = self.conn_.cursor()

        # Try exact part number match first
        cur.execute(
            "SELECT * FROM spare_parts WHERE part_number = ?",
            (part_number,)
        )
        rows = cur.fetchall()

        if not rows:
            # Try as asset_id → find compatible parts
            cur.execute(
                "SELECT * FROM spare_parts WHERE compatible_assets LIKE ?",
                (f"%{part_number}%",)
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            part = dict(row)
            available = part["qty_on_hand"] - part["qty_reserved"]
            part["qty_available"] = available
            part["in_stock"] = available > 0
            part["below_reorder"] = part["qty_on_hand"] <= part["reorder_level"]
            results.append(part)

        if not results:
            return [{"error": f"No parts found for '{part_number}'."}]
        return results

    def get_maintenance_summary(self, asset_id: str) -> dict:
        """
        Complete maintenance context for the RCA agent.

        Combines:
            - Asset information
            - Recent work order history
            - Compatible spare parts availability

        This is the primary method called by the RCA agent.
        Returns a structured dict ready for LLM prompt injection.
        """
        if not self.is_initialized_:
            raise RuntimeError("Call initialize() first.")

        asset_id = self._resolve_asset_id(asset_id)

        asset_info   = self.get_asset_info(asset_id)
        work_orders  = self.get_work_orders(asset_id, n=5)
        spare_parts  = self.check_spare_parts(asset_id)

        # Compute time since last maintenance
        last_wo_date = None
        if work_orders:
            last_wo_date = work_orders[0].get("completed_date")

        days_since_last_wo = None
        if last_wo_date:
            last_dt = datetime.strptime(last_wo_date, "%Y-%m-%d")
            days_since_last_wo = (datetime(2004, 2, 15) - last_dt).days

        # Build formatted summary for LLM
        summary_text = self._format_summary(
            asset_info, work_orders, spare_parts, days_since_last_wo
        )

        return {
            "asset":              asset_info,
            "work_orders":        work_orders,
            "spare_parts":        spare_parts,
            "days_since_last_wo": days_since_last_wo,
            "summary_text":       summary_text,
        }

    def _format_summary(self, asset: dict, work_orders: list,
                         spare_parts: list,
                         days_since_last: int | None) -> str:
        """
        Format maintenance summary as text for the RCA agent prompt.
        """
        lines = [
            "=== CMMS Maintenance Summary ===",
            "",
            f"Asset: {asset.get('asset_id', 'N/A')} — "
            f"{asset.get('description', 'N/A')}",
            f"Model: {asset.get('model', 'N/A')}",
            f"Installed: {asset.get('install_date', 'N/A')} "
            f"({asset.get('days_in_service', '?')} days in service)",
            f"Location: {asset.get('location', 'N/A')}",
            f"Criticality: {asset.get('criticality', 'N/A')}",
            f"Rated speed: {asset.get('rated_speed', '?')} RPM",
            "",
        ]

        if days_since_last is not None:
            lines.append(
                f"Days since last maintenance: {days_since_last}")
        lines.append(f"Total work orders on file: {len(work_orders)}")
        lines.append("")

        # Recent work orders
        lines.append("Recent Work Orders:")
        for wo in work_orders[:3]:
            lines.append(
                f"  [{wo['wo_number']}] {wo['completed_date']} "
                f"({wo['wo_type']}/{wo['priority']})")
            lines.append(f"    {wo['description']}")
            lines.append(f"    Findings: {wo['findings'][:200]}")
            lines.append(f"    Actions: {wo['actions_taken'][:200]}")
            lines.append("")

        # Spare parts
        lines.append("Spare Parts Availability:")
        for part in spare_parts:
            if "error" in part:
                lines.append(f"  {part['error']}")
            else:
                status = ("IN STOCK" if part["in_stock"]
                          else "OUT OF STOCK")
                reorder = (" ⚠ BELOW REORDER LEVEL"
                           if part.get("below_reorder") else "")
                lines.append(
                    f"  {part['part_number']}: {part['description']}")
                lines.append(
                    f"    {status} — {part['qty_available']} available "
                    f"(of {part['qty_on_hand']} on hand){reorder}")
                lines.append(
                    f"    Location: {part['location']} | "
                    f"Lead time: {part['lead_time_days']} days")

        return "\n".join(lines)

    # ── Utility ───────────────────────────────────────────────────────────

    def execute_query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute arbitrary SQL query (for advanced agent use)."""
        cur = self.conn_.cursor()
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def close(self):
        """Close database connection."""
        if self.conn_:
            self.conn_.close()

    def stats(self) -> dict:
        """Return database statistics."""
        cur = self.conn_.cursor()
        return {
            "assets":      cur.execute(
                "SELECT COUNT(*) FROM assets").fetchone()[0],
            "work_orders": cur.execute(
                "SELECT COUNT(*) FROM work_orders").fetchone()[0],
            "spare_parts": cur.execute(
                "SELECT COUNT(*) FROM spare_parts").fetchone()[0],
            "db_path":     self.db_path,
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("CMMS MCP Server — BearingMind")
    print("=" * 60)

    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/cmms.db"

    cmms = CMMSMCP(db_path=db_path)
    cmms.initialize(force_rebuild=True)

    print(f"\nStats: {cmms.stats()}")

    # Demo queries — what the RCA agent would ask
    print("\n" + "─" * 60)
    print("Asset Info — BRG-001 (also accessible via 'b1_ch1'):")
    info = cmms.get_asset_info("b1_ch1")
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n" + "─" * 60)
    print("Work Orders — BRG-001:")
    for wo in cmms.get_work_orders("BRG-001"):
        print(f"  [{wo['wo_number']}] {wo['completed_date']} "
              f"({wo['wo_type']}) — {wo['description'][:60]}")

    print("\n" + "─" * 60)
    print("Spare Parts — compatible with BRG-001:")
    for part in cmms.check_spare_parts("BRG-001"):
        if "error" not in part:
            print(f"  {part['part_number']}: {part['qty_available']} "
                  f"available | {'IN STOCK' if part['in_stock'] else 'OUT'}")

    print("\n" + "─" * 60)
    print("Full Maintenance Summary — BRG-004:")
    summary = cmms.get_maintenance_summary("BRG-004")
    print(summary["summary_text"])

    cmms.close()
