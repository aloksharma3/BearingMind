# BearingMind

**Multi-agent predictive maintenance system for rotating machinery — from raw vibration signals to actionable maintenance decisions.**

BearingMind detects bearing faults, predicts remaining useful life, explains *why* the fault is happening using SHAP, then queries equipment manuals and maintenance databases to produce a structured Root Cause Analysis report with specific parts, actions, and urgency levels.

Built on the NASA IMS Bearing Dataset. Designed to demonstrate the architecture used in production systems at companies like Schneider Electric, Siemens, ABB, and GE Vernova.

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
git clone https://github.com/yourusername/bearingmind.git
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

# Step 3: Train LSTM RUL predictors
python src/rul_lstm.py results/feature_matrix.csv results/rul/ 30 50

# Step 4: Convert signals to spectrogram images
python src/signal_to_image.py <path-to-2nd_test> data/images/ 500 4

# Step 5: Train CNN Autoencoder on healthy spectrograms
python src/cv_anomaly_detector.py data/images/normal data/images/all results/cv/ 50

# Step 6: Run SHAP explainability (uses saved models)
python src/shap_explainer.py results/feature_matrix.csv results/ results/shap/ 950
```

## Feature engineering

16 features extracted per bearing per snapshot:

- **Time-domain**: RMS, peak-to-peak, kurtosis, crest factor, skewness, shape factor, impulse factor, margin factor
- **Frequency-domain**: spectral centroid, spectral bandwidth, spectral entropy, dominant frequency, high-frequency energy ratio
- **Fault-band energy**: BPFO (outer race), BPFI (inner race), BSF (ball)

## Explainability

The SHAP explainer provides three layers of evidence for the RCA agent:

1. **Isolation Forest SHAP** (TreeExplainer) — which statistical features drive the anomaly score
2. **LSTM SHAP** (GradientExplainer) — which features drive the RUL prediction
3. **CNN reconstruction heatmap** — which frequency bands in the spectrogram show abnormal patterns, mapped to fault types via frequency → fault mapping

Example RCA context output:
```
Bearing: b1_ch1
Snapshot index: 950
Anomaly score: 0.4523
RUL score: 0.1200 (CRITICAL)

Top anomaly drivers (Isolation Forest SHAP):
  - BPFO band energy (outer race): +0.3100 (↑ toward anomaly)
  - Kurtosis: +0.2400 (↑ toward anomaly)

CNN Autoencoder spectral analysis:
  Reconstruction MSE: 0.012345 (ANOMALY)
  Worst channel: STFT spectrogram
  Worst frequency band: mid_low
  Band fault inference: primary bearing fault frequency (BPFO/BPFI/BSF range)

Probable fault type: outer race fault
```

## Repository structure

```
industrial-ai-scada/
├── src/
│   ├── __init__.py
│   ├── features.py              # 16-feature extraction
│   ├── isolation_forest.py      # Isolation Forest anomaly detection
│   ├── signal_to_image.py       # STFT + Mel + GAF → 64×64×3 images
│   ├── cv_anomaly_detector.py   # CNN Autoencoder anomaly detection
│   ├── rul_lstm.py              # LSTM RUL prediction
│   └── shap_explainer.py        # SHAP + CNN heatmaps → RCA context
├── notebooks/
│   ├── nasa_bearing_eda.ipynb
│   └── nasa_bearing_eda.py
├── results/                     # Generated outputs (see .gitignore)
├── requirements.txt
├── .gitignore
└── README.md
```

## Next steps (Week 2)

- LangGraph multi-agent orchestration
- MCP servers: equipment manual RAG, CMMS mock, weather API, parts inventory
- RCA Agent using Claude API for plain-English diagnostic reports
- Streamlit or Gradio dashboard for live monitoring

## Author

**Alok** — MS student at Northeastern University. 6 years of industry experience in industrial IoT, SCADA systems, and enterprise automation (L&T, Schneider Electric). Building at the intersection of industrial OT domain knowledge and AI/ML.

---

## License

This project is for educational and portfolio purposes. The NASA IMS Bearing Dataset is publicly available from NASA's Prognostics Center of Excellence.