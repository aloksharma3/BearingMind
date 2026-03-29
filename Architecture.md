# BearingMind — Technical Architecture

This document describes the design decisions, data flow, and component interactions in detail. Read the README first for the overview.

---

## Design principles

**Model-agnostic components.** Each module exposes a consistent interface. The LSTM can be swapped for TFT or Chronos-2 without touching SHAP or the RCA agent. The TF-IDF retriever can be swapped for sentence-transformers without touching the RCA agent. The MCP tool interface is the contract — internals are implementation details.

**Per-bearing models.** Each of the 4 bearings gets its own Isolation Forest and its own LSTM. Bearings degrade differently depending on position and load. A single model across all bearings would average out the signals.

**Train on healthy data only.** The Isolation Forest and CNN Autoencoder are trained exclusively on the first 500 snapshots (known healthy period). This matches production practice — you rarely have labeled failure data for the specific asset you're monitoring.

**Explain before acting.** The RCA agent never receives raw ML scores. It receives SHAP-attributed evidence with specific feature contributions and inferred fault types. This is the trust layer that makes the system usable by human engineers.

---

## Data flow

```
NASA IMS raw data (20 kHz, 4 channels)
    │
    ▼
features.py
    │  Reads raw vibration files
    │  Computes 16 features per bearing per snapshot:
    │    Time-domain:  RMS, peak-to-peak, kurtosis, crest factor,
    │                  skewness, shape factor, impulse factor, margin factor
    │    Frequency:    spectral centroid, spectral bandwidth, spectral entropy,
    │                  dominant frequency, HF energy ratio
    │    Fault bands:  BPFO band energy, BPFI band energy, BSF band energy
    │  Output: feature_matrix.csv (984 rows × 64 columns)
    │
    ├──────────────────────────────┐
    ▼                              ▼
isolation_forest.py           rul_lstm.py
    │                              │
    │  Train on rows 0-499         │  Train on all rows with
    │  (healthy period)            │  linear RUL labels (1→0)
    │  Score all 984 rows          │  Window size: 30 snapshots
    │  Threshold: 99th pct         │  Architecture: LSTM(128)→LSTM(64)→Dense(32)→1
    │  Output: anomaly scores      │  Output: RUL predictions [0.0, 1.0]
    │  + binary flags              │
    │                              │
    └──────────┬───────────────────┘
               ▼
         shap_explainer.py
               │
               │  TreeExplainer on Isolation Forest
               │    → exact SHAP values per feature
               │  GradientExplainer on LSTM
               │    → backprop-based SHAP values (averaged over window)
               │  Fault inference: top SHAP features → fault type
               │    BPFO band energy → outer race fault
               │    BPFI band energy → inner race fault
               │    BSF band energy  → ball fault
               │    kurtosis         → impulsive fault
               │  Output: rca_context dict (evidence + fault type)
               │
               ▼
         rca_agent.py
               │
               │  1. Receives SHAP context
               │  2. Looks up fault type in FAULT_QUERY_MAP
               │  3. Sends queries to Equipment Manual MCP
               │  4. Sends asset query to CMMS MCP
               │  5. Synthesizes report (Claude API or template)
               │
               ├──▶ mcp_equipment_manual.py
               │      TF-IDF vectorize query
               │      Cosine similarity against 12 chunks
               │      Return top-k passages with source/section
               │
               ├──▶ mcp_cmms.py
               │      Translate bearing_id → asset_id
               │      SQL query: asset info + work orders + spare parts
               │      Return formatted maintenance summary
               │
               ▼
         RCA Report
               │  Diagnosis (fault type)
               │  Evidence (SHAP values, anomaly/RUL scores)
               │  Probable root cause (manual passages)
               │  Maintenance history (CMMS work orders)
               │  Recommended actions (fault-specific, parts-aware)
               │  Urgency level (CRITICAL/HIGH/MEDIUM/LOW)
               │  Parts required (with stock status and location)
```

---

## Feature engineering details

The 16 features per bearing are chosen to cover three detection strategies:

**Time-domain features** detect changes in signal amplitude and distribution. RMS tracks overall vibration energy. Kurtosis and crest factor detect impulsive events (sharp spikes from spalling). These catch faults early — before spectral methods can see them.

**Spectral features** detect changes in frequency content. Spectral centroid shifts when new fault frequencies appear. Spectral entropy increases when energy spreads from narrow peaks to broadband noise (late-stage damage). HF energy ratio catches surface degradation that pushes energy above the normal operating band.

**Fault band energies** are the diagnostic core. BPFO, BPFI, and BSF are the mathematically computed bearing characteristic frequencies based on bearing geometry (number of rolling elements, pitch diameter, contact angle, ball diameter). Energy concentrated at these frequencies is direct evidence of a specific fault location. This is what makes the SHAP explanation actionable — "BPFI energy is the top contributor" directly translates to "inner race fault."

---

## Model design decisions

**Why Isolation Forest over One-Class SVM?** Isolation Forest handles high-dimensional data (16 features) without kernel tricks, trains faster, and produces calibrated anomaly scores. It also naturally handles the feature correlations in bearing data without explicit preprocessing.

**Why LSTM over Transformer for RUL?** The NASA IMS dataset has 984 time steps — too small for a Transformer to learn meaningful attention patterns. LSTM's inductive bias (sequential processing, forget gate) is well-suited to monotonic degradation curves. The architecture is also directly defensible in interviews — it's the standard choice in published IMS benchmark papers. TFT or Chronos-2 can be swapped in later as model-agnostic upgrades.

**Why CNN Autoencoder alongside Isolation Forest?** They detect different failure modes. Isolation Forest works on engineered features (structured). The CNN Autoencoder works on STFT spectrograms (images) — it catches spatial patterns in the frequency domain that scalar features might miss. In production, you'd run both and take the union of their alerts.

**Why TF-IDF over embedding models for manual RAG?** Zero external dependencies. For 12 chunks from 3 manuals, TF-IDF with bigrams gives precision comparable to dense embeddings. The MCP interface is identical regardless of retrieval backend — swapping to sentence-transformers or OpenAI embeddings is a one-file change.

---

## MCP server design

Both MCP servers follow the same pattern:

1. **TOOL_SCHEMA** — JSON schema describing the tool name, description, and parameters. This is what gets registered with LangGraph (or any MCP client) so the LLM knows what tools are available.

2. **Stateful initialization** — `load()` for the manual server (builds TF-IDF index), `initialize()` for the CMMS server (creates SQLite tables). Separated from `__init__` so the object can be created before the graph starts and initialized when the graph is ready.

3. **Tool methods** — the actual functions the agent calls. Each returns a dict or list of dicts. Errors are returned as dicts with an `error` key, not as exceptions — defensive design for agent use where you don't want an exception to crash the whole graph.

4. **ID resolution** — the CMMS server translates between bearing IDs (b1_ch1, used by ML) and asset IDs (BRG-001, used by CMMS). This mapping layer is a real-world pattern — sensor systems and maintenance systems almost never share the same identifiers.

---

## RCA agent decision logic

The agent's behavior changes based on the detected fault type:

| Fault type | Manual queries | Key actions |
|---|---|---|
| Inner race (BPFI) | BPFI diagnosis, lubrication/spalling | Check grease, verify alignment |
| Outer race (BPFO) | BPFO diagnosis, housing inspection | Verify housing bore, inspect seals |
| Ball fault (BSF) | BSF detection, contamination/EDM | Flush lubricant, check for electrical pitting |
| Impulsive (kurtosis) | Kurtosis/spalling, replacement procedure | Spectrum analysis to confirm location |
| Surface degradation | HF energy/wear, failure stages | Immediate action — late stage damage |
| Distributed damage | Spectral entropy, failure model | Emergency assessment — multiple modes |

Urgency is determined by the RUL score: ≤0.10 = CRITICAL, ≤0.25 = HIGH, ≤0.50 = MEDIUM, >0.50 = LOW.

Parts recommendations are dynamic — the agent checks actual stock levels from the CMMS and either confirms availability (with warehouse location) or flags an emergency procurement need with lead time.

---

## Planned: LangGraph orchestration (Week 3)

The current pipeline is sequential. LangGraph will add conditional routing:

```
                    ┌──────────────┐
                    │  Fault       │
                    │  Detection   │
                    └──────┬───────┘
                           │
                     ┌─────┴─────┐
                     │ anomaly?  │
                     └─────┬─────┘
                    yes    │    no
                   ┌───────┴───────┐
                   ▼               ▼
            ┌────────────┐  ┌──────────┐
            │ SHAP + RUL │  │   Log    │
            │ analysis   │  │ & exit   │
            └─────┬──────┘  └──────────┘
                  │
            ┌─────┴─────┐
            │ urgency?  │
            └─────┬─────┘
         CRITICAL │    HIGH/MEDIUM
          ┌───────┴───────┐
          ▼               ▼
   ┌────────────┐  ┌────────────┐
   │ Alert      │  │ RCA Agent  │
   │ Agent      │  │ + MCP      │
   │ (immediate │  │ (standard  │
   │  notify)   │  │  report)   │
   └────────────┘  └────────────┘
```

This is what makes it agentic — the system decides its own next step based on intermediate results. Low anomaly scores skip expensive SHAP computation entirely. Critical urgency triggers immediate alerts instead of waiting for a full report.

---

## File-by-file summary

| File | Lines | Purpose |
|---|---|---|
| `features.py` | ~250 | 16-feature extraction per bearing from raw vibration |
| `isolation_forest.py` | ~300 | Per-bearing unsupervised anomaly detection |
| `rul_lstm.py` | ~400 | Stacked LSTM for remaining useful life prediction |
| `cv_anomaly_detector.py` | ~250 | CNN autoencoder on STFT spectrograms |
| `signal_to_image.py` | ~200 | STFT, Mel spectrogram, GAF image conversion |
| `shap_explainer.py` | ~500 | SHAP for both models + fault type inference |
| `mcp_equipment_manual.py` | ~300 | TF-IDF RAG over maintenance manuals |
| `mcp_cmms.py` | ~450 | SQLite CMMS with assets, work orders, parts |
| `rca_agent.py` | ~830 | RCA agent wiring MCP tools + report generation |