# BearingMind

**Multi-agent predictive maintenance system for rotating machinery — from raw vibration signals to actionable maintenance decisions, with a live explainability dashboard.**

BearingMind detects bearing faults, predicts remaining useful life, explains *why* the fault is happening using SHAP, then routes the evidence through a chain of AI agents that query equipment manuals, maintenance databases, and live weather conditions to produce a structured Root Cause Analysis report — with specific parts, actions, and urgency levels displayed in a real-time React dashboard.

---

## What it does

A vibration sensor on a bearing produces 20,000 samples per second. BearingMind processes that data through a four-node agentic pipeline:

| Node | Agent | Output |
|------|-------|--------|
| DETECT | Isolation Forest + LSTM | Anomaly score, RUL %, system status |
| EXPLAIN | SHAP TreeExplainer | Top features driving the anomaly, fault type inference |
| DIAGNOSE | RCA Agent (3 MCP tools) | Structured fault report with evidence, history, parts |
| ALERT | Alert Agent | Notification routing, CMMS work order creation |

The output is a report that tells a maintenance engineer: *"Inner race fault detected on Bearing 2. BPFI band energy is the primary driver (SHAP +0.29). Per the Vibration Diagnostics Handbook, check lubrication first — it causes 60% of inner race failures. Last maintenance was 118 days ago. Replacement bearing RX-ZA2115 is in stock (3 available, Warehouse A, Shelf B3-07). Schedule within 48 hours."*

All of this is visible live in the dashboard — select any snapshot, click Analyze Live, and watch the pipeline run end-to-end.

---

## Architecture

```
Vibration Data (NASA IMS — 984 snapshots, 4 bearings, Feb 12–19 2004)
        │
        ▼
┌─────────────────┐
│  features.py    │  16 features × 4 bearings (RMS, kurtosis, BPFO/BPFI/BSF...)
└────────┬────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌────────┐  ┌────────┐
│  ISO   │  │  LSTM  │  Anomaly score + flag / RUL 0→1
│ FOREST │  │  RUL   │
└───┬────┘  └───┬────┘
    └─────┬─────┘
          ▼
┌─────────────────┐
│ SHAP Explainer  │  Per-feature importance, fault type inference
│ (TreeExplainer) │  Positive SHAP = pushes toward anomaly
└────────┬────────┘
         │          Only runs when anomaly_count > 0 OR RUL < 30%
         ▼
┌─────────────────┐     ┌──────────────────────────┐
│   RCA Agent     │────▶│  Equipment Manual MCP     │  TF-IDF RAG
│                 │     │  (SKF, Vibration Guide)   │  12 chunks, 3 sources
│                 │     └──────────────────────────┘
│                 │     ┌──────────────────────────┐
│                 │────▶│  CMMS MCP (SQLite)        │  4 assets, 8 WOs, 5 parts
│                 │     │  assets / WOs / parts     │
│                 │     └──────────────────────────┘
│                 │     ┌──────────────────────────┐
│                 │────▶│  Weather MCP (Open-Meteo) │  Live temp, humidity, risk
└────────┬────────┘     └──────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Alert Agent    │  Routes by urgency: SMS/email/Slack/dashboard
│                 │  Auto-creates CMMS work orders for HIGH/CRITICAL
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  React Dashboard  (FastAPI backend) │
│  DETECT → EXPLAIN → DIAGNOSE → ALERT│
└─────────────────────────────────────┘
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
│   ├── mcp_weather.py              Weather MCP server (Open-Meteo API)
│   ├── rca_agent.py                Root Cause Analysis agent
│   ├── alert_agent.py              Alert & notification routing agent
│   └── orchestrator.py             Multi-agent orchestrator
│
├── api/
│   └── server.py                   FastAPI backend — live inference + static serving
│
├── dashboard/
│   ├── src/
│   │   └── App.jsx                 React dashboard (Recharts, Vite)
│   ├── dist/                       Compiled frontend (served by FastAPI)
│   └── package.json
│
├── results/
│   ├── feature_matrix.csv          984×64 extracted features (committed)
│   ├── if/
│   │   ├── models/                 Saved .pkl detectors per bearing
│   │   └── anomaly_scores.csv      Historical scores (committed)
│   ├── rul/
│   │   ├── models/                 Saved .pt LSTM models per bearing
│   │   ├── rul_predictions.csv     Historical predictions (committed)
│   │   └── rul_metrics.csv         MAE/RMSE per bearing (committed)
│   ├── shap/                       SHAP waterfall + summary plots
│   └── rca/                        Alert log, RCA reports
│
├── notebooks/
│   └── nasa_bearing_eda.ipynb      Exploratory data analysis
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/aloksharma3/BearingMind.git
cd BearingMind
pip install -r requirements.txt
```

### 2. Download the NASA IMS dataset

Download the 2nd test dataset from the [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) and place the `.txt` snapshot files in `data/IMS/`.

### 3. Run the ML pipeline (one-time)

Pre-computed CSVs and model checkpoints are committed, so this step is only needed if you want to retrain from scratch.

```bash
cd src

# Extract features (generates results/feature_matrix.csv)
python features.py

# Train anomaly detector
python isolation_forest.py ../results/feature_matrix.csv ../results/if

# Train RUL predictor
python rul_lstm.py ../results/feature_matrix.csv ../results/rul

# Run SHAP explainability
python shap_explainer.py ../results/feature_matrix.csv ../results ../results/shap
```

### 4. Start the dashboard

```bash
# Start the FastAPI backend (serves API + compiled dashboard)
python api/server.py
```

- Use the slider to select any snapshot (0–983)
- Click **Analyze Live** to run the full DETECT → EXPLAIN → DIAGNOSE → ALERT pipeline
- The agentic path in the topbar lights up as each node completes

### 5. (Optional) Enable LLM-generated RCA reports

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
python api/server.py
```

Without the API key, the RCA agent runs in template mode — same MCP queries, same structured output, same pipeline. With the key, Claude generates richer natural language analysis in the Diagnosis section.

### 6. (Optional) Rebuild the frontend after editing App.jsx

```bash
cd dashboard
npm install
npm run build
```

---

## Dataset

**NASA IMS Bearing Dataset — 2nd Test**

Four Rexnord ZA-2115 double row bearings on a loaded shaft running at 2000 RPM. PCB 353B33 accelerometers sampled at 20 kHz. 984 snapshots (~10 min each) over 7 days of continuous operation.

| Property | Value |
|---|---|
| Bearings | 4 (b1_ch1 through b4_ch1) |
| Sampling rate | 20 kHz |
| Snapshots | 984 |
| Duration | Feb 12–19, 2004 |
| Known failure | Outer race defect — Bearing 1 |

> **Note on ground truth vs model output:** The NASA IMS dataset confirms B1 (drive-end) developed an outer race fault. The Isolation Forest ranks B4 (free-end) highest at snapshot 950 — this reflects early multi-bearing degradation before the B1 failure becomes dominant. Both are correct; they reflect different evidence windows.

---

## ML models

### Isolation Forest (anomaly detection)

Unsupervised detector trained on the first 500 snapshots (healthy period). One model per bearing, scored independently against its healthy baseline. Threshold at 99th percentile of training scores.

### LSTM (remaining useful life)

Two-layer stacked LSTM (128 → 64 units) with dropout. Trained on linear RUL labels (1.0 → 0.0). Window size of 30 snapshots. Status thresholds:

| RUL | Status |
|-----|--------|
| ≥ 0.30 | NORMAL |
| 0.15 – 0.29 | WARNING |
| < 0.15 | CRITICAL |

Per-bearing accuracy:

| Bearing | MAE | RMSE |
|---------|-----|------|
| b1_ch1 (Drive-End) | 0.093 | 0.121 |
| b2_ch1 (Inner-1) | 0.063 | 0.078 |
| b3_ch1 (Inner-2) | 0.050 | 0.062 |
| b4_ch1 (Free-End) | 0.177 | 0.226 |

### CNN Autoencoder (visual anomaly detection)

Convolutional autoencoder trained on STFT spectrograms (64×64×3) of healthy vibration signals. Detects anomalies via reconstruction error. Per-channel heatmaps map high-error frequency bands back to fault types (BPFO, BPFI, BSF).

---

## MCP servers

Three MCP-style tool servers give the RCA agent structured access to external knowledge:

### Equipment Manual MCP

RAG server over bearing maintenance manuals. 12 knowledge base chunks from 3 sources (SKF Bearing Guide, Vibration Diagnostics Handbook, Industrial Motor Manual). TF-IDF retrieval with bigram indexing. The agent queries it based on the SHAP-inferred fault type — different fault types trigger different manual queries.

### CMMS MCP

Simulated Computerized Maintenance Management System backed by SQLite. Three tables: `assets` (4 bearings), `work_orders` (8 maintenance records going back to 2003), `spare_parts` (5 items with live stock levels). Exposes 4 tools: `get_asset_info`, `get_work_orders`, `check_spare_parts`, `get_maintenance_summary`.

Note: SQLite connection uses `check_same_thread=False` so it works correctly when called from FastAPI's thread pool.

### Weather MCP

Fetches live conditions from the [Open-Meteo](https://open-meteo.com) API for Cincinnati, OH (the NASA IMS test site). Maps temperature and humidity to a combined environmental risk level (LOW / MEDIUM / HIGH). Displayed in the dashboard's Environment card and included in the RCA report.

---

## SHAP explainability

Every prediction is explained:

- **TreeExplainer** for Isolation Forest — exact SHAP values for all 16 features
- **GradientExplainer** for LSTM — backpropagation-based SHAP averaged over the 30-snapshot window
- **Sign convention:** IF `decision_function` is positive-for-normal, so raw SHAP values are negated at the API layer. Positive SHAP = pushes toward anomaly (red in dashboard). Negative SHAP = pushes toward normal (green).
- **Fault inference:** top SHAP features map to fault types (BPFO band → outer race, BPFI band → inner race, BSF band → ball fault, kurtosis → impulsive/spalling)

---

## Alert routing

The Alert Agent routes RCA reports by urgency level:

| Urgency | Channels | Work order |
|---------|----------|------------|
| CRITICAL | SMS + email (supervisor, plant manager) + Slack #critical-alerts | EMERGENCY |
| HIGH | Email (planner, reliability eng) + Slack #maintenance | HIGH priority |
| MEDIUM | Email (planner) | None |
| LOW | Dashboard log only | None |

RCA and alerting only run when there is a real problem: `anomaly_count > 0` OR `min_rul < 0.30`. Healthy snapshots show a green "All bearings healthy" state instead of running the full pipeline.

---

## Dashboard

The React dashboard shows:

- **Topbar** — agentic pipeline path (DETECT → EXPLAIN → DIAGNOSE → ALERT), lights up as nodes execute
- **Stat cards** — most critical bearing (lowest RUL), highest anomaly score, system status, anomaly count
- **Bearing health grid** — RUL % per bearing with color-coded progress bars (red/amber/green by proximity to failure)
- **Time series charts** — anomaly scores and RUL predictions across all 984 snapshots with real calendar dates (Feb 12–19) and bearing color legends
- **RCA panel** — full diagnosis, recommended actions, CMMS work order history, spare parts inventory, equipment manual RAG results, weather impact, alert routing summary
- **SHAP panel** — all 16 features ranked by importance with +/− bars (positive = anomaly-driving)
- **Bottom row** — dataset/model info with ground truth note, LSTM accuracy (MAE/RMSE per bearing), live environment conditions

---

## Roadmap

-  Feature extraction (16 features × 4 bearings)
-  Isolation Forest anomaly detection
-  LSTM remaining useful life prediction
-  CNN autoencoder (visual anomaly detection)
-  SHAP explainability layer (sign-corrected for IF)
-  Equipment Manual MCP server (TF-IDF RAG)
-  CMMS MCP server (SQLite)
-  Weather MCP (Open-Meteo live conditions)
-  RCA agent with 3-tool MCP integration
-  Alert agent (urgency-based routing + CMMS work order creation)
-  FastAPI backend with live inference endpoint
-  React dashboard with real-time SHAP, RCA, and agentic pipeline visualization
-  False-positive guard (RCA gated on actual anomalies)
-  LangGraph orchestrator (conditional agent routing with graph visualization)
---

## Tech stack

| Component | Technology |
|---|---|
| Feature engineering | NumPy, SciPy |
| Anomaly detection | scikit-learn (Isolation Forest) |
| RUL prediction | PyTorch (LSTM) |
| Visual anomaly detection | TensorFlow/Keras (CNN Autoencoder) |
| Explainability | SHAP (TreeExplainer + GradientExplainer) |
| Manual retrieval | scikit-learn (TF-IDF), cosine similarity |
| Maintenance database | SQLite |
| Weather data | Open-Meteo API |
| RCA agent | Anthropic Claude API (optional), template fallback |
| Alert routing | Rule-based agent with CMMS integration |
| Backend API | FastAPI + Uvicorn |
| Dashboard | React 18 + Recharts + Vite |

---

## Author

**Alok** — MS student at Northeastern University. 6 years of industry experience in industrial IoT, SCADA systems, and enterprise automation (L&T, Schneider Electric). Building at the intersection of AI/ML and industrial OT domain knowledge.

---

## License

This project is for educational and portfolio purposes. The NASA IMS Bearing Dataset is publicly available from NASA's Prognostics Center of Excellence.
