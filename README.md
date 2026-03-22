# BearingMind — Industrial AI Predictive Maintenance

**Multi-model bearing fault detection using vibration analysis, deep learning, and explainable AI**

## What this project does

BearingMind monitors industrial bearings using vibration sensor data and answers three questions a maintenance engineer needs:

1. **Is something wrong?** — Isolation Forest on statistical features + CNN Autoencoder on spectrograms
2. **How much life is left?** — LSTM neural network predicting Remaining Useful Life (RUL)
3. **Why does the model think that?** — SHAP explainability + CNN reconstruction error heatmaps

All three answers feed into an RCA (Root Cause Analysis) context that a downstream LLM agent can use to generate plain-English maintenance reports.

## Architecture

```
NASA IMS raw vibration data (20,480 Hz × 4 bearings × 984 snapshots)
    │
    ├── features.py ──────────────── 16 statistical features per bearing
    │       │
    │       ├── isolation_forest.py ─ unsupervised anomaly detection
    │       │
    │       └── rul_lstm.py ──────── LSTM RUL prediction (0→1 degradation)
    │
    ├── signal_to_image.py ──────── STFT + Mel + GAF → 64×64×3 images
    │       │
    │       └── cv_anomaly_detector.py ── CNN Autoencoder anomaly detection
    │
    └── shap_explainer.py ──────── SHAP (IF + LSTM) + CNN heatmaps → RCA context
```

## Models

| Module | Model | Task | Input |
|---|---|---|---|
| `isolation_forest.py` | Isolation Forest | Anomaly detection | 16 statistical features |
| `rul_lstm.py` | 2-layer stacked LSTM | RUL prediction | Sliding window of features |
| `cv_anomaly_detector.py` | CNN Autoencoder | Visual anomaly detection | 64×64×3 spectrograms |
| `shap_explainer.py` | SHAP + heatmaps | Explainability | All model outputs |

## Dataset

**NASA IMS Bearing Dataset** — 2nd test set

- 4 bearings monitored simultaneously at 20,480 Hz
- 984 snapshots over ~35 days of continuous operation
- Run-to-failure: bearing 3 develops an outer race fault
- Download: [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

## Quick start

```bash
git clone https://github.com/<your-username>/industrial-ai-scada.git
cd industrial-ai-scada
pip install -r requirements.txt
```

### Run the full pipeline

```bash
# Step 1: Extract features from raw vibration data
python src/features.py <path-to-2nd_test> results/feature_matrix.csv

# Step 2: Train Isolation Forest anomaly detectors
python src/isolation_forest.py results/feature_matrix.csv results/if/ 500

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

**Alok** — MS student at Northeastern University, focusing on Industrial AI and predictive maintenance for energy and manufacturing systems. 6 years of prior experience in industrial IoT and SCADA systems.
