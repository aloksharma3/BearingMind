"""
mcp_equipment_manual.py — Equipment Manual MCP Server
Industrial AI Predictive Maintenance | BearingMind

An MCP (Model Context Protocol) server that exposes equipment manuals
as a queryable tool for the RCA agent.

What this does:
    The RCA agent detects "inner race fault" via SHAP → it needs to know
    what the maintenance manual says about inner race faults. This server
    answers that question by searching through chunked manual text.

How it works:
    1. On startup, loads manual documents from a knowledge base directory
    2. Chunks them into overlapping passages (~300 words each)
    3. Builds a TF-IDF index over all chunks
    4. Exposes a `query_manual` tool that returns top-k relevant passages

Why TF-IDF instead of vector embeddings:
    - Zero external dependencies (no OpenAI/Cohere API key needed)
    - Fast enough for a portfolio project with <100 pages of manuals
    - Easily swappable to sentence-transformers or OpenAI embeddings later
    - The MCP interface stays the same regardless of retrieval backend

MCP tool exposed:
    query_manual(query: str, top_k: int = 3) → list[dict]
        Returns top-k most relevant manual passages for the query.
        Each result has: source, section, text, relevance_score

Knowledge base structure:
    data/manuals/
    ├── skf_bearing_maintenance.json    (chunked SKF bearing manual)
    ├── vibration_diagnostics.json       (vibration analysis guide)
    └── ... (add more .json files to expand coverage)

Pipeline position:
    SHAP explainer → RCA Agent → [query_manual] → Equipment Manual MCP
                                                    ↓
                                              relevant passages
                                                    ↓
                                              RCA Agent uses them
                                              in fault report

Usage:
    from src.mcp_equipment_manual import EquipmentManualMCP

    mcp = EquipmentManualMCP(knowledge_base_dir="data/manuals")
    mcp.load()

    results = mcp.query_manual(
        "What maintenance actions are recommended for inner race faults?",
        top_k=3
    )
    for r in results:
        print(f"[{r['source']}] {r['section']}")
        print(f"  {r['text'][:200]}...")
        print(f"  relevance: {r['relevance_score']:.3f}")
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional

# ── TF-IDF for retrieval ──────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Manual chunk schema ───────────────────────────────────────────────────────

def _default_knowledge_base() -> list[dict]:
    """
    Built-in equipment manual knowledge base.

    In production this would be loaded from PDFs via OCR + chunking.
    For the portfolio project, we embed realistic manual content directly.
    This covers the fault types that SHAP actually identifies in the
    NASA IMS dataset (outer race, inner race, ball fault, general wear).

    Each entry simulates a chunk from a real maintenance manual.
    Content is based on publicly available SKF and FAG bearing
    maintenance guidelines — rewritten to avoid copyright issues.
    """
    return [
        # ── SKF Bearing Maintenance Guide ─────────────────────────────────
        {
            "source": "SKF Bearing Maintenance Guide",
            "section": "Chapter 3.1 — Inner Race Fault Diagnosis",
            "text": (
                "Inner race faults produce characteristic vibration patterns "
                "at the Ball Pass Frequency Inner race (BPFI). The BPFI "
                "signature appears as periodic impulses modulated by the "
                "shaft rotation frequency. Early-stage inner race defects "
                "cause elevated kurtosis in the time-domain signal before "
                "any audible change is detectable. Recommended actions upon "
                "BPFI detection: (1) Verify lubrication condition — "
                "inadequate grease is the primary cause of inner race "
                "spalling in 60% of premature failures. (2) Check shaft "
                "alignment and interference fit — excessive preload "
                "accelerates inner race fatigue. (3) Schedule bearing "
                "replacement within the next planned maintenance window "
                "if RUL estimate is below 30%. Do not operate beyond "
                "detectable spalling — catastrophic seizure risk increases "
                "exponentially once inner race cracks propagate."
            ),
        },
        {
            "source": "SKF Bearing Maintenance Guide",
            "section": "Chapter 3.2 — Outer Race Fault Diagnosis",
            "text": (
                "Outer race defects generate vibration at the Ball Pass "
                "Frequency Outer race (BPFO). Unlike inner race faults, "
                "BPFO signatures remain relatively stable in amplitude "
                "because the defect is stationary relative to the sensor "
                "mounting point. Common root causes include: (1) Improper "
                "mounting — the bearing housing bore must be within "
                "tolerance to prevent race distortion. (2) Static "
                "overloading during transport or installation. (3) "
                "Corrosion from moisture ingress in humid environments. "
                "Maintenance response: verify housing fit, inspect seals "
                "for moisture damage, and check bearing temperature trend. "
                "If outer race BPFO energy exceeds 3x baseline, schedule "
                "immediate replacement. Outer race faults progress slower "
                "than inner race faults but can cause secondary damage "
                "to the cage and rolling elements if left unaddressed."
            ),
        },
        {
            "source": "SKF Bearing Maintenance Guide",
            "section": "Chapter 3.3 — Rolling Element (Ball) Fault",
            "text": (
                "Ball spin frequency (BSF) defects are among the hardest "
                "to detect because the defect rotates in and out of the "
                "load zone. This produces an intermittent vibration "
                "signature that modulates with cage rotation frequency. "
                "Elevated BSF band energy combined with high spectral "
                "entropy indicates distributed surface damage across "
                "multiple rolling elements. Root causes: (1) Lubricant "
                "contamination — particles as small as 5 microns can "
                "initiate surface pitting. (2) Electrical discharge "
                "machining (EDM) from stray current in VFD-driven motors. "
                "(3) Material fatigue from extended operation beyond rated "
                "load. Corrective actions: flush and replace lubricant, "
                "inspect for electrical pitting marks (frosting pattern), "
                "and verify load calculations. Replace bearing if BSF "
                "harmonics are visible at 2x and 3x BSF frequency."
            ),
        },
        {
            "source": "SKF Bearing Maintenance Guide",
            "section": "Chapter 4.1 — Lubrication Schedules",
            "text": (
                "Bearing lubrication intervals depend on speed, load, and "
                "operating temperature. For standard deep groove ball "
                "bearings (6200 series) at moderate speeds (<3600 RPM): "
                "re-grease every 4000 operating hours using lithium-complex "
                "grease NLGI Grade 2. Symptoms of under-lubrication: "
                "rising RMS vibration, increasing bearing temperature, "
                "elevated high-frequency energy ratio. Symptoms of "
                "over-lubrication: elevated temperature without vibration "
                "increase, grease leakage from seals. Over-lubrication "
                "accounts for 36% of grease-related bearing failures in "
                "industrial applications. Use calculated grease volume: "
                "G = 0.005 × D × B (where D = outside diameter in mm, "
                "B = bearing width in mm). Always verify grease "
                "compatibility before mixing brands."
            ),
        },
        {
            "source": "SKF Bearing Maintenance Guide",
            "section": "Chapter 5.2 — Bearing Replacement Procedure",
            "text": (
                "Standard bearing replacement for 6205-series deep groove "
                "ball bearings: (1) Lock out / tag out the drive system. "
                "(2) Remove coupling guard and disconnect motor coupling. "
                "(3) Use induction heater to heat replacement bearing to "
                "110°C — never use open flame. (4) Slide bearing onto shaft "
                "with steady pressure on inner ring. Never apply force to "
                "outer ring. (5) Allow to cool and verify axial clearance "
                "with feeler gauge (target: 0.05-0.10mm for C3 fit). "
                "(6) Pack grease to 30-50% of free volume in bearing "
                "cavity. (7) Reassemble housing, torque bolts to spec. "
                "(8) Run motor unloaded for 15 minutes, verify temperature "
                "stabilizes below 70°C. Record all measurements in CMMS "
                "work order for baseline comparison."
            ),
        },

        # ── Vibration Analysis Reference Guide ────────────────────────────
        {
            "source": "Vibration Diagnostics Handbook",
            "section": "Section 2.1 — Time-Domain Indicators",
            "text": (
                "Kurtosis is the most sensitive early indicator of bearing "
                "damage. A healthy bearing produces vibration with Gaussian "
                "distribution (kurtosis ≈ 3.0). Incipient spalling creates "
                "sharp impulses that raise kurtosis above 4.0. Values "
                "above 6.0 indicate definite damage. Crest factor (peak / "
                "RMS) above 6.0 similarly indicates impulsive events. "
                "In early degradation stages, kurtosis rises before RMS "
                "because the impulses are sharp but infrequent. As damage "
                "progresses, RMS rises and kurtosis may actually decrease "
                "as the impulses become continuous. This is why a single "
                "indicator is unreliable — the SHAP-based multi-feature "
                "approach captures the full degradation trajectory."
            ),
        },
        {
            "source": "Vibration Diagnostics Handbook",
            "section": "Section 2.3 — Spectral Entropy for Fault Severity",
            "text": (
                "Spectral entropy quantifies the disorder in the frequency "
                "spectrum. A healthy bearing has low spectral entropy — "
                "energy concentrated at shaft and bearing characteristic "
                "frequencies. As damage spreads, energy distributes across "
                "broader frequency bands, raising entropy. High spectral "
                "entropy combined with elevated high-frequency energy ratio "
                "indicates distributed surface degradation affecting "
                "multiple components simultaneously. This condition is "
                "more severe than a single-point defect and warrants "
                "immediate maintenance action. The bearing is likely in "
                "Stage 3 or Stage 4 of the four-stage failure model."
            ),
        },
        {
            "source": "Vibration Diagnostics Handbook",
            "section": "Section 3.1 — Four-Stage Bearing Failure Model",
            "text": (
                "Stage 1 (Subsurface origin): Microscopic cracks form "
                "below the raceway surface. No detectable vibration change. "
                "Detectable only by ultrasonic emission (>250 kHz). "
                "Stage 2 (Defect initiation): Cracks reach surface, "
                "creating small spalls. BPFI/BPFO/BSF frequencies appear. "
                "Kurtosis rises. RMS may still be normal. This is the "
                "optimal detection and intervention window. "
                "Stage 3 (Defect propagation): Spalls grow. RMS rises "
                "visibly. Multiple bearing frequencies and their harmonics "
                "become prominent. Temperature begins to increase. "
                "Schedule immediate replacement. "
                "Stage 4 (Catastrophic failure): Bearing geometry "
                "distorts. Broadband vibration dominates. High risk of "
                "seizure, shaft damage, and collateral equipment damage. "
                "Emergency shutdown required."
            ),
        },
        {
            "source": "Vibration Diagnostics Handbook",
            "section": "Section 4.2 — Sensor Placement Guidelines",
            "text": (
                "Accelerometer placement critically affects fault detection "
                "sensitivity. Mount sensors on the bearing housing in the "
                "load zone direction — typically vertical for horizontal "
                "shaft machines. Use stud mounting for permanent "
                "installations (frequency response up to 10 kHz). Magnetic "
                "mounts are acceptable for route-based monitoring but "
                "reduce usable bandwidth to ~5 kHz. Ensure flat, clean "
                "mounting surface — paint and rust under the sensor can "
                "reduce sensitivity by 50%. For the NASA IMS test rig, "
                "sensors were PCB 353B33 accelerometers sampled at "
                "20 kHz, mounted directly on the bearing housing, "
                "providing bandwidth sufficient for all bearing "
                "characteristic frequencies up to Nyquist (10 kHz)."
            ),
        },

        # ── Motor Maintenance Manual ──────────────────────────────────────
        {
            "source": "Industrial Motor Maintenance Manual",
            "section": "Section 6.1 — Motor Bearing Life Calculation",
            "text": (
                "Basic bearing life L10 (in millions of revolutions) is "
                "calculated as L10 = (C/P)^3 for ball bearings, where "
                "C = dynamic load rating (from manufacturer catalog) and "
                "P = equivalent dynamic bearing load. For a 6205 bearing "
                "(C = 14.8 kN) under 2 kN load at 1800 RPM, L10 = "
                "403 million revolutions or approximately 3,700 hours. "
                "Adjusted life L10a accounts for lubrication quality "
                "(factor a2) and contamination (factor a3). In clean, "
                "well-lubricated conditions, actual life can be 5-10x "
                "the basic L10 rating. In contaminated environments, "
                "life may be reduced to 10-20% of L10. The LSTM RUL "
                "predictor provides a data-driven alternative to "
                "L10 calculations by learning actual degradation "
                "patterns from vibration features."
            ),
        },
        {
            "source": "Industrial Motor Maintenance Manual",
            "section": "Section 6.3 — Root Cause Analysis Checklist",
            "text": (
                "When a bearing fault is detected, investigate these "
                "root causes in order: (1) Lubrication — is grease "
                "quantity and quality correct? Check re-greasing log in "
                "CMMS. (2) Alignment — measure angular and parallel "
                "misalignment. Laser alignment should show <0.05mm "
                "offset. (3) Balance — check for unbalance at 1x RPM "
                "frequency in spectrum. (4) Load — verify actual load "
                "vs. bearing rating. (5) Installation — was bearing "
                "properly fitted? Check shaft and housing tolerances. "
                "(6) Contamination — inspect seals, check environment "
                "for dust/moisture. (7) Electrical — for VFD-driven "
                "motors, check shaft grounding and bearing insulation. "
                "Document findings and corrective actions in CMMS work "
                "order for pattern analysis."
            ),
        },
        {
            "source": "Industrial Motor Maintenance Manual",
            "section": "Section 7.1 — Spare Parts Inventory Management",
            "text": (
                "Critical spare parts for rotating equipment should "
                "follow a tiered stocking strategy. Tier 1 (keep on-site): "
                "bearings, seals, and lubricant for all critical assets. "
                "For the bearing test rig, this includes SKF 6205-2RS "
                "deep groove ball bearings and Regal-Beloit coupling "
                "elements. Tier 2 (regional warehouse, 24-hour delivery): "
                "motor windings, shaft sleeves, and coupling assemblies. "
                "Tier 3 (manufacturer lead time): complete motor "
                "assemblies, custom housings. Minimum stock levels should "
                "be set based on criticality and mean time to failure. "
                "Parts inventory should be integrated with condition "
                "monitoring — when RUL drops below 30%, automatically "
                "verify spare parts availability and raise purchase "
                "requisition if below minimum stock."
            ),
        },
    ]


# ── Equipment Manual MCP Server ───────────────────────────────────────────────

class EquipmentManualMCP:
    """
    MCP server for equipment manual retrieval.

    Loads manual chunks → builds TF-IDF index → exposes query_manual tool.

    The RCA agent calls this tool when it needs maintenance context
    for a detected fault. For example:
        - SHAP says "BPFI band energy" is the top contributor
        - RCA agent infers "inner race fault"
        - RCA agent calls: query_manual("inner race fault maintenance")
        - This server returns relevant manual passages about inner race
          fault diagnosis, root causes, and corrective actions

    Args:
        knowledge_base_dir : path to directory with manual JSON files
                             (optional — uses built-in KB if not provided)
    """

    # ── MCP tool schema (for LangGraph tool registration) ─────────────────
    TOOL_SCHEMA = {
        "name": "query_manual",
        "description": (
            "Search equipment maintenance manuals for information about "
            "bearing faults, maintenance procedures, vibration diagnostics, "
            "and root cause analysis. Use this when you need technical "
            "context about a detected fault type or maintenance action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language query about the fault or "
                        "maintenance topic. Examples: 'inner race fault "
                        "maintenance actions', 'BPFO diagnosis procedure', "
                        "'bearing lubrication schedule'"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of relevant passages to return",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    }

    def __init__(self, knowledge_base_dir: Optional[str] = None):
        self.knowledge_base_dir = knowledge_base_dir
        self.chunks_: list[dict] = []
        self.vectorizer_: TfidfVectorizer | None = None
        self.tfidf_matrix_ = None
        self.is_loaded_ = False

    def load(self) -> "EquipmentManualMCP":
        """
        Load knowledge base and build TF-IDF index.

        Loading strategy:
            1. If knowledge_base_dir is provided AND contains .json files,
               load chunks from those files
            2. Otherwise, use the built-in knowledge base
            3. Build TF-IDF matrix over all chunks

        JSON format expected:
            [{"source": "...", "section": "...", "text": "..."}, ...]
        """
        # Load chunks
        if self.knowledge_base_dir and os.path.isdir(self.knowledge_base_dir):
            json_files = list(Path(self.knowledge_base_dir).glob("*.json"))
            if json_files:
                self.chunks_ = []
                for jf in sorted(json_files):
                    with open(jf) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        self.chunks_.extend(data)
                    print(f"  Loaded {len(data)} chunks from {jf.name}")
            else:
                print("  No JSON files found — using built-in knowledge base")
                self.chunks_ = _default_knowledge_base()
        else:
            self.chunks_ = _default_knowledge_base()

        print(f"  Knowledge base: {len(self.chunks_)} chunks from "
              f"{len(set(c['source'] for c in self.chunks_))} sources")

        # Build TF-IDF index
        corpus = [
            f"{c['section']} {c['text']}" for c in self.chunks_
        ]
        self.vectorizer_ = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),      # unigrams + bigrams for phrase matching
            sublinear_tf=True,       # dampens term frequency
        )
        self.tfidf_matrix_ = self.vectorizer_.fit_transform(corpus)

        self.is_loaded_ = True
        print(f"  TF-IDF index built: {self.tfidf_matrix_.shape[1]} features")
        return self

    # ── MCP tool implementation ───────────────────────────────────────────

    def query_manual(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search the equipment manual knowledge base.

        This is the method the RCA agent calls via MCP.

        Args:
            query  : natural language query about a fault or procedure
            top_k  : number of results to return (default 3)

        Returns:
            List of dicts, each with:
                - source          : manual name (e.g. "SKF Bearing Guide")
                - section         : chapter/section heading
                - text            : full passage text
                - relevance_score : cosine similarity (0-1)
        """
        if not self.is_loaded_:
            raise RuntimeError("Call load() before querying.")

        # Vectorize query
        q_vec = self.vectorizer_.transform([query])

        # Cosine similarity against all chunks
        scores = cosine_similarity(q_vec, self.tfidf_matrix_).flatten()

        # Top-k results
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] > 0.0:  # only return actual matches
                chunk = self.chunks_[idx]
                results.append({
                    "source":          chunk["source"],
                    "section":         chunk["section"],
                    "text":            chunk["text"],
                    "relevance_score": round(float(scores[idx]), 4),
                })

        return results

    # ── Utility methods ───────────────────────────────────────────────────

    def list_sources(self) -> list[str]:
        """Return list of all manual source names in the knowledge base."""
        return sorted(set(c["source"] for c in self.chunks_))

    def list_sections(self, source: str = None) -> list[str]:
        """Return all section headings, optionally filtered by source."""
        if source:
            return [c["section"] for c in self.chunks_
                    if c["source"] == source]
        return [c["section"] for c in self.chunks_]

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Add new chunks and rebuild the TF-IDF index.
        Use this to expand the knowledge base at runtime.
        """
        self.chunks_.extend(chunks)
        # Rebuild index
        corpus = [f"{c['section']} {c['text']}" for c in self.chunks_]
        self.tfidf_matrix_ = self.vectorizer_.fit_transform(corpus)
        print(f"  Index rebuilt: {len(self.chunks_)} chunks, "
              f"{self.tfidf_matrix_.shape[1]} features")

    def stats(self) -> dict:
        """Return knowledge base statistics."""
        sources = set(c["source"] for c in self.chunks_)
        return {
            "total_chunks": len(self.chunks_),
            "sources": sorted(sources),
            "n_sources": len(sources),
            "tfidf_features": (self.tfidf_matrix_.shape[1]
                               if self.tfidf_matrix_ is not None else 0),
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Equipment Manual MCP Server — BearingMind")
    print("=" * 60)

    kb_dir = sys.argv[1] if len(sys.argv) > 1 else None

    # Initialize and load
    mcp = EquipmentManualMCP(knowledge_base_dir=kb_dir)
    mcp.load()

    print(f"\nStats: {mcp.stats()}")
    print(f"\nSources: {mcp.list_sources()}")

    # Demo queries — these are what the RCA agent would actually ask
    demo_queries = [
        "inner race fault diagnosis and maintenance actions",
        "outer race BPFO vibration bearing replacement",
        "ball fault rolling element BSF detection",
        "kurtosis crest factor early stage bearing damage",
        "lubrication schedule grease interval",
        "bearing remaining useful life L10 calculation",
        "root cause analysis checklist for bearing failure",
    ]

    for query in demo_queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        results = mcp.query_manual(query, top_k=2)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r['source']} → {r['section']}")
            print(f"      Score: {r['relevance_score']:.3f}")
            print(f"      {r['text'][:120]}...")
