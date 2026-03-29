# BearingMind

**Multi-agent predictive maintenance system for rotating machinery — from raw vibration signals to actionable maintenance decisions.**

BearingMind detects bearing faults, predicts remaining useful life, explains *why* the fault is happening using SHAP, then queries equipment manuals and maintenance databases to produce a structured Root Cause Analysis report with specific parts, actions, and urgency levels.

Designed to demonstrate the architecture used in production systems at companies like Schneider Electric, Siemens, ABB, and GE Vernova.

---

## What it does

A vibration sensor on a bearing produces 20,000 samples per second. BearingMind processes that data through a pipeline of ML models and AI agents:

1. **Feature extraction** — 16 engineered features per bearing (RMS, kurtosis, BPFO/BPFI/BSF fault band energies, spectral indicators)
2. **Anomaly detection** — Isolation Forest trained on healthy data flags when a bearing deviates from normal
3. **Remaining Useful Life prediction** — Stacked LSTM estimates how much life remains (0% to 100%)
4. **Explainability** — SHAP identifies which specific features are driving the anomaly and RUL predictions
5. **Root Cause Analysis** — An AI agent queries equipment manuals (RAG) and the CMMS database (SQLite), then synthesizes a structured maintenance report

The output is a report that tells a maintenance engineer: *"Inner race fault detected on Bearing 2. BPFI band energy is the primary driver. Per the SKF manual, check lubrication first — it causes 60% of inner race failures. Last maintenance was 118 days ago. Replacement bearing RX-ZA2115 is in stock (3 available, Warehouse A, Shelf B3-07). Schedule within 48 hours."*

---

## Architecture

```
Vibration Data (NASA IMS)
        │
        ▼
┌─────────────────┐
│  features.py    │  16 features × 4 bearings × 984 snapshots
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ Iso.   │ │ LSTM   │  Anomaly detection + RUL prediction
│ Forest │ │ RUL    │
└───┬────┘ └───┬────┘
    │          │
    └────┬─────┘
         ▼
┌─────────────────┐
│ SHAP Explainer  │  Top features + fault type inference
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────┐
│   RCA Agent     │────▶│ Equipment Manual MCP  │  TF-IDF RAG
│                 │     │ (SKF, vibration guide)│
│                 │     └──────────────────────┘
│                 │     ┌──────────────────────┐
│                 │────▶│ CMMS MCP (SQLite)     │  Work orders, parts
│                 │     │ (assets, inventory)   │
└────────┬────────┘     └──────────────────────┘
         │
         ▼
┌─────────────────┐
│  RCA Report     │  Diagnosis, evidence, actions, parts, urgency
└─────────────────┘
```

---

## Project structure

```
bearingmind/
├── src/
│   ├── features.py                 Feature extraction (16 per bearing)
│   ├── isolation_forest.py         Unsupervised anomaly detection
│   ├── rul_lstm.py                 LSTM remaining useful life predictor
│   ├── cv_anomaly_detector.py      CNN autoencoder anomaly detector
│   ├── signal_to_image.py          STFT / Mel / GAF image conversion
│   ├── shap_explainer.py           SHAP explainability layer
│   ├── mcp_equipment_manual.py     Equipment Manual MCP server (RAG)
│   ├── mcp_cmms.py                 CMMS MCP server (SQLite)
│   └── rca_agent.py                Root Cause Analysis agent
│
├── results/
│   ├── feature_matrix.csv          984×64 extracted features
│   ├── if/                         Isolation Forest outputs
│   │   ├── models/                 Saved .pkl detectors per bearing
│   │   ├── anomaly_scores.csv
│   │   └── anomaly_plot.png
│   ├── rul/                        LSTM RUL outputs
│   │   ├── models/                 Saved .pt models per bearing
│   │   ├── rul_predictions.csv
│   │   └── rul_plot.png
│   ├── shap/                       SHAP explanations
│   │   ├── shap_report.csv
│   │   └── waterfall + summary plots
│   └── rca/                        RCA agent outputs
│       ├── rca_report.txt
│       └── rca_metadata.json
│
├── notebooks/
│   └── nasa_bearing_eda.ipynb      Exploratory data analysis
│
├── .gitignore
├── requirements.txt
├── ARCHITECTURE.md                 Detailed technical architecture
└── README.md
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/aloksharma3/BearingMind.git
cd bearingmind
pip install -r requirements.txt
```

### 2. Download the NASA IMS dataset

Download the 2nd test dataset from the [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) and place it in `data/IMS/`.

### 3. Run the pipeline

```bash
# Step 1: Extract features (run once — generates results/feature_matrix.csv)
cd src
python features.py

# Step 2: Train anomaly detector
python isolation_forest.py ../results/feature_matrix.csv ../results/if

# Step 3: Train RUL predictor
python rul_lstm.py ../results/feature_matrix.csv ../results/rul

# Step 4: Run SHAP explainability
python shap_explainer.py ../results/feature_matrix.csv ../results ../results/shap

# Step 5: Run full RCA pipeline (loads trained models, queries MCP servers)
python rca_agent.py ../results/feature_matrix.csv ../results 950 ../results/rca
```

### 4. (Optional) Enable LLM-generated reports

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
python rca_agent.py ../results/feature_matrix.csv ../results 950 ../results/rca
```

Without the API key, the RCA agent works in template mode — same MCP queries, same structure, same pipeline. With the API key, Claude generates richer natural language analysis.

---

## Dataset

**NASA IMS Bearing Dataset — 2nd Test**

Four Rexnord ZA-2115 double row bearings on a loaded shaft running at 2000 RPM. PCB 353B33 accelerometers sampled at 20 kHz. 984 snapshots over ~7 days of continuous operation until outer race failure on Bearing 1.

| Property | Value |
|---|---|
| Bearings | 4 (b1_ch1 through b4_ch1) |
| Sampling rate | 20 kHz |
| Snapshots | 984 |
| Duration | ~7 days |
| Failure mode | Outer race defect (Bearing 1) |

---

## ML models

### Isolation Forest (anomaly detection)

Unsupervised detector trained on the first 500 snapshots (healthy period). Scores each bearing independently against its healthy baseline. Threshold at 99th percentile of training scores.

### LSTM (remaining useful life)

Two-layer stacked LSTM (128 → 64 units) with dropout. Trained on linear RUL labels (1.0 → 0.0). Window size of 30 snapshots. Predicts a normalized RUL score where values below 0.15 are flagged as CRITICAL.

| Bearing | MAE | RMSE |
|---|---|---|
| b1_ch1 | 0.093 | 0.121 |
| b2_ch1 | 0.063 | 0.078 |
| b3_ch1 | 0.050 | 0.062 |
| b4_ch1 | 0.177 | 0.226 |

### CNN Autoencoder (visual anomaly detection)

Convolutional autoencoder trained on STFT spectrograms (64×64×3) of healthy vibration signals. Detects anomalies via reconstruction error.

---

## MCP servers

### Equipment Manual MCP

RAG server over bearing maintenance manuals. 12 knowledge base chunks from 3 sources (SKF Bearing Guide, Vibration Diagnostics Handbook, Industrial Motor Manual). TF-IDF retrieval with bigram indexing. Covers all fault types identified by SHAP: inner race, outer race, ball fault, impulsive damage, surface degradation, lubrication, and replacement procedures.

### CMMS MCP (SQLite)

Simulated Computerized Maintenance Management System backed by SQLite. Three tables: `assets` (4 bearings), `work_orders` (8 maintenance records), `spare_parts` (5 items with stock levels). Exposes 4 tools: `get_asset_info`, `get_work_orders`, `check_spare_parts`, `get_maintenance_summary`.

---

## SHAP explainability

Every prediction is explained by SHAP:

- **TreeExplainer** for Isolation Forest — exact SHAP values for each of the 16 features
- **GradientExplainer** for LSTM — backpropagation-based SHAP values averaged over the prediction window
- **Fault inference** — maps top SHAP features to fault types (BPFO → outer race, BPFI → inner race, BSF → ball fault)

The SHAP output is the bridge between the ML models and the RCA agent. Without it, the agent would be guessing. With it, the agent has evidence to cite.

---

## Example output

Running `python rca_agent.py ../results/feature_matrix.csv ../results 950 ../results/rca` produces:

```
DIAGNOSIS: INNER RACE FAULT detected on b2_ch1
URGENCY: HIGH

EVIDENCE:
  Anomaly score: 0.1407
  RUL score: 0.1614 (WARNING)
  Top driver: BPFI band energy (inner race)

MAINTENANCE HISTORY:
  Days since last maintenance: 118
  Last finding: "BRG-002 stable. RMS 0.40 mm/s. No concerns."

RECOMMENDED ACTIONS:
  1. Schedule emergency maintenance within 48 hours
  2. Inspect lubrication condition (60% of inner race failures)
  3. Verify shaft alignment and interference fit
  4. Replacement bearing RX-ZA2115 in stock (3 available)

PARTS: RX-ZA2115, MOBIL-SHC220-1KG, SEAL-ZA2115-V — all in stock
```

---

## Roadmap

- [x] Feature extraction (16 features × 4 bearings)
- [x] Isolation Forest anomaly detection
- [x] LSTM remaining useful life prediction
- [x] CNN autoencoder (visual anomaly detection)
- [x] SHAP explainability layer
- [x] Equipment Manual MCP server (RAG)
- [x] CMMS MCP server (SQLite)
- [x] RCA agent with MCP tool integration
- [ ] Weather MCP (ambient temperature impact on bearing life)
- [ ] Alert agent (notification routing by urgency)
- [ ] LangGraph orchestrator (conditional agent routing)
- [ ] Streamlit dashboard
- [ ] Demo video

---

## Tech stack

| Component | Technology |
|---|---|
| Feature engineering | NumPy, SciPy, librosa |
| Anomaly detection | scikit-learn (Isolation Forest) |
| RUL prediction | PyTorch (LSTM) |
| Visual anomaly detection | PyTorch (CNN Autoencoder) |
| Explainability | SHAP |
| Manual retrieval | scikit-learn (TF-IDF), cosine similarity |
| Maintenance database | SQLite |
| RCA agent | Claude API (optional), template fallback |
| Orchestration | LangGraph|
| Dashboard | Streamlit |

---

## Author

**Alok** — MS student at Northeastern University. 6 years of industry experience in industrial IoT, SCADA systems, and enterprise automation (L&T, Schneider Electric). Building at the intersection of industrial OT domain knowledge and AI/ML.

---

## License

This project is for educational and portfolio purposes. The NASA IMS Bearing Dataset is publicly available from NASA's Prognostics Center of Excellence.
