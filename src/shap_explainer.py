"""
shap_explainer.py — SHAP Explainability Layer
Industrial AI Predictive Maintenance | BearingMind

Wraps explainers around all three trained models:
    - BearingAnomalyDetector  (Isolation Forest) → SHAP TreeExplainer
    - BearingRULPredictor     (LSTM)             → SHAP GradientExplainer
    - CVAnomalyDetector       (CNN Autoencoder)  → Reconstruction Error Heatmap

For the CNN Autoencoder, we don't use SHAP directly — the anomaly score
is reconstruction MSE, not a single forward-pass prediction. Instead we
compute a per-channel reconstruction error heatmap that shows WHERE the
spectrogram looks abnormal. We then map those spatial regions back to
frequency bands → fault-type indicators (BPFO, BPFI, BSF).

Why heatmaps instead of SHAP for the CNN?
    SHAP for autoencoders is expensive and hard to interpret — the model
    has 64×64×3 = 12,288 input features, and the "prediction" is an MSE
    reduction, not a class label. Reconstruction error heatmaps are:
    1. Standard practice in published papers (ABB 2025, Siemens 2024)
    2. Directly interpretable — bright regions = abnormal
    3. Mappable to fault frequencies via the STFT axis layout
    4. Fast — no background set or gradient computation needed

For every anomaly flagged or low RUL predicted, the explainer tells you:
    "kurtosis contributed +0.31, bpfo_band_energy contributed +0.24 ..."
    "CNN heatmap shows high error in STFT channel at mid-frequency band ..."

This is the bridge between raw ML scores and the RCA Agent (LLM).
Without explainability the RCA Agent is guessing. With it, evidence to cite.

Output per bearing per snapshot:
    - SHAP values for each of the 16 features (IF + LSTM)
    - CNN reconstruction error heatmap + top-error channel + frequency band
    - Top-3 contributing features with direction (↑ anomaly / ↓ normal)
    - Ready-to-use context dict for the RCA Agent prompt

Pipeline position:
    isolation_forest.py    ──┐
                              ├─→ shap_explainer.py → RCA Agent
    rul_lstm.py            ──┤
                              │
    cv_anomaly_detector.py ──┘

Usage:
    from src.shap_explainer import BearingShapExplainer

    explainer = BearingShapExplainer(
        anomaly_detector, rul_predictor, cv_detector=cv_detector)
    explainer.fit(feature_matrix_df)

    # Explain a single snapshot (for real-time agent use)
    context = explainer.explain_snapshot(snapshot_index=950)

    # Explain all anomalous snapshots (for batch analysis)
    report_df = explainer.explain_anomalies(anomaly_scores_df)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── SHAP import ────────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Install with: pip install shap")

# ── PyTorch import ─────────────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── TensorFlow import (for CNN Autoencoder explainability) ─────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from sklearn.preprocessing import StandardScaler

# ── Feature name mapping ───────────────────────────────────────────────────────
# Maps raw feature column names to human-readable labels for plots and reports
# Matches exactly the 16 features produced by features.py
FEATURE_LABELS = {
    "rms":               "RMS vibration",
    "peak_to_peak":      "Peak-to-peak amplitude",
    "kurtosis":          "Kurtosis",
    "crest_factor":      "Crest factor",
    "skewness":          "Skewness",
    "shape_factor":      "Shape factor",
    "impulse_factor":    "Impulse factor",
    "margin_factor":     "Margin factor",
    "spectral_centroid": "Spectral centroid (Hz)",
    "spectral_bandwidth":"Spectral bandwidth (Hz)",
    "spectral_entropy":  "Spectral entropy",
    "dominant_freq_hz":  "Dominant frequency (Hz)",
    "hf_energy_ratio":   "High-freq energy ratio",
    "bpfo_band_energy":  "BPFO band energy (outer race)",
    "bpfi_band_energy":  "BPFI band energy (inner race)",
    "bsf_band_energy":   "BSF band energy (ball fault)",
}

# Fault type inference from top SHAP features
# Used by explain_snapshot() to suggest a probable fault type
FAULT_INFERENCE_MAP = {
    "bpfo_band_energy":  "outer race fault",
    "bpfi_band_energy":  "inner race fault",
    "bsf_band_energy":   "rolling element (ball) fault",
    "kurtosis":          "impulsive fault (spalling / cracking)",
    "crest_factor":      "impulsive fault (early stage)",
    "hf_energy_ratio":   "surface degradation (broadband wear)",
    "spectral_entropy":  "distributed damage (multiple fault modes)",
    "margin_factor":     "early-stage damage",
}

# ── CNN Heatmap frequency band mapping ────────────────────────────────────────
# The 64×64 spectrogram image has frequency on the vertical axis (row 0 = highest).
# We divide the 64 rows into 4 frequency bands for the RCA agent.
# NASA IMS sampling rate = 20,480 Hz → Nyquist = 10,240 Hz
# STFT with nperseg=256 → 129 frequency bins → resized to 64 rows
CNN_IMAGE_SIZE = 64
CNN_FREQ_BANDS = {
    "high_freq":   (0,  15),   # rows 0-15   → ~6400-10240 Hz (harmonics, HF noise)
    "mid_high":    (16, 31),   # rows 16-31  → ~2560-6400 Hz  (bearing fault freqs)
    "mid_low":     (32, 47),   # rows 32-47  → ~1280-2560 Hz  (BPFO/BPFI range)
    "low_freq":    (48, 63),   # rows 48-63  → ~0-1280 Hz     (shaft speed, 1X, 2X)
}

# Map CNN frequency bands to probable fault types
CNN_BAND_FAULT_MAP = {
    "high_freq":  "surface degradation (broadband HF energy)",
    "mid_high":   "bearing fault harmonics (2×BPFO, 2×BPFI)",
    "mid_low":    "primary bearing fault frequency (BPFO/BPFI/BSF range)",
    "low_freq":   "imbalance or misalignment (shaft speed harmonics)",
}

# Channel names for the 3-channel RGB spectrogram image
CNN_CHANNEL_NAMES = ["STFT spectrogram", "Mel spectrogram", "Gramian angular field"]


# ── Single Bearing SHAP Explainer ──────────────────────────────────────────────

class SingleBearingShapExplainer:
    """
    Explainer for one bearing — wraps all three models.

    Holds:
        - TreeExplainer         for the Isolation Forest detector
        - GradientExplainer     for the LSTM RUL predictor
        - Reconstruction heatmap for the CNN Autoencoder (optional)

    Args:
        bearing_id         : e.g. 'b1_ch1'
        anomaly_detector   : fitted SingleBearingDetector instance
        rul_predictor      : fitted SingleBearingRUL instance
        cv_detector        : fitted CVAnomalyDetector instance (optional)
    """

    def __init__(self, bearing_id: str,
                 anomaly_detector,
                 rul_predictor,
                 cv_detector=None):
        self.bearing_id       = bearing_id
        self.anomaly_detector = anomaly_detector
        self.rul_predictor    = rul_predictor
        self.cv_detector      = cv_detector
        self.tree_explainer_  = None
        self.grad_explainer_  = None
        self.background_X_    = None   # scaled features for background
        self.feature_names_   = None
        self.is_fitted_       = False

    def fit(self, df: pd.DataFrame,
            n_background: int = 100) -> "SingleBearingShapExplainer":
        """
        Initialise SHAP explainers using training data as background.

        Background data = first n_background snapshots (healthy period).
        SHAP uses background to compute baseline expected output,
        then measures how much each feature shifts predictions away from it.

        Args:
            df            : full feature matrix DataFrame
            n_background  : number of background samples (default 100)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP required. pip install shap")

        cols = self.anomaly_detector.feature_cols_
        self.feature_names_ = cols

        # Scale features using the anomaly detector's scaler
        X_raw    = df[cols].values
        X_scaled = self.anomaly_detector.scaler.transform(X_raw)

        # Background = first n_background healthy snapshots
        bg = X_scaled[:n_background]
        self.background_X_ = bg

        # ── Tree Explainer (Isolation Forest) ─────────────────────────────
        # TreeExplainer is exact and fast for tree-based models.
        # We explain the decision_function output (anomaly score).
        self.tree_explainer_ = shap.TreeExplainer(
            self.anomaly_detector.model,
            data=bg,
            feature_perturbation="interventional",
        )

        # ── Gradient Explainer (LSTM) ──────────────────────────────────────
        # GradientExplainer uses backpropagation to compute SHAP values.
        # We take the last snapshot in each window as the "current" input.
        if TORCH_AVAILABLE and self.rul_predictor.is_fitted_:
            X_rul_raw    = df[self.rul_predictor.feature_cols_].values
            X_rul_scaled = self.rul_predictor.scaler_.transform(
                X_rul_raw).astype(np.float32)

            window = self.rul_predictor.window_size

            # Build background windows from healthy snapshots
            bg_windows = []
            for i in range(min(n_background, len(X_rul_scaled) - window)):
                bg_windows.append(X_rul_scaled[i: i + window])
            bg_tensor = torch.tensor(
                np.stack(bg_windows), dtype=torch.float32)

            self.rul_predictor.model_.eval()
            self.grad_explainer_ = shap.GradientExplainer(
                self.rul_predictor.model_,
                bg_tensor.to(self.rul_predictor.device_),
            )

        self.is_fitted_ = True
        return self

    # ── Core explain methods ──────────────────────────────────────────────────

    def explain_anomaly_score(self, df: pd.DataFrame,
                               snapshot_idx: int) -> dict:
        """
        SHAP values for the Isolation Forest anomaly score
        at a specific snapshot.

        Returns dict:
            {
              'shap_values'  : np.ndarray (n_features,),
              'feature_names': list of feature column names,
              'feature_labels': list of human-readable labels,
              'top_features' : list of top-3 (feature, shap_val, direction),
              'base_value'   : explainer expected value,
              'anomaly_score': raw model score at this snapshot
            }
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")

        cols     = self.feature_names_
        X_raw    = df[cols].values
        X_scaled = self.anomaly_detector.scaler.transform(X_raw)
        x        = X_scaled[snapshot_idx: snapshot_idx + 1]

        shap_vals = self.tree_explainer_.shap_values(x)[0]  # (n_features,)
        anomaly_score = float(
            -self.anomaly_detector.model.decision_function(x)[0])

        top_features = self._top_features(
            shap_vals, cols, n=3, invert=True)  # invert: higher = more anomalous

        return {
            "shap_values":    shap_vals,
            "feature_names":  cols,
            "feature_labels": [self._label(c) for c in cols],
            "top_features":   top_features,
            "base_value":     float(self.tree_explainer_.expected_value),
            "anomaly_score":  anomaly_score,
        }

    def explain_rul(self, df: pd.DataFrame,
                    snapshot_idx: int) -> dict:
        """
        SHAP values for the LSTM RUL prediction at a specific snapshot.

        Returns dict:
            {
              'shap_values'      : np.ndarray (n_features,) — averaged over window,
              'feature_names'    : list,
              'feature_labels'   : list,
              'top_features'     : list of top-3 (feature, shap_val, direction),
              'rul_score'        : predicted RUL at this snapshot
            }
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")
        if self.grad_explainer_ is None:
            return {"error": "LSTM SHAP explainer not initialised — "
                    "check PyTorch is installed and LSTM is fitted."}

        rul_cols  = self.rul_predictor.feature_cols_
        X_raw     = df[rul_cols].values
        X_scaled  = self.rul_predictor.scaler_.transform(
            X_raw).astype(np.float32)
        window    = self.rul_predictor.window_size

        start = max(0, snapshot_idx - window)
        x_win = X_scaled[start: snapshot_idx]

        # Pad if window is shorter than expected
        if len(x_win) < window:
            pad   = np.zeros((window - len(x_win), x_win.shape[1]),
                             dtype=np.float32)
            x_win = np.vstack([pad, x_win])

        x_tensor = torch.tensor(
            x_win[np.newaxis], dtype=torch.float32
        ).to(self.rul_predictor.device_)

        shap_vals_win = self.grad_explainer_.shap_values(
            x_tensor)   # (1, window, n_features)

        # GradientExplainer with (batch,1) output returns a list of length=n_outputs
        # Each element has shape (batch, window, features)
        # Unwrap completely to a clean 2D (window, features) array
        if isinstance(shap_vals_win, list):
            # list of tensors/arrays — one per output node
            raw = shap_vals_win[0]
        else:
            raw = shap_vals_win

        # Convert tensor to numpy if needed
        if hasattr(raw, 'detach'):
            raw = raw.detach().cpu().numpy()
        raw = np.array(raw, dtype=np.float32)

        # Squeeze out all size-1 dims, then ensure 2D (window, features)
        raw = raw.squeeze()          # removes batch=1 and output=1 dims
        if raw.ndim == 1:
            # edge case: single timestep — make it (1, features)
            raw = raw[np.newaxis, :]
        elif raw.ndim > 2:
            # still has extra dims — flatten to (window, features)
            raw = raw.reshape(-1, raw.shape[-1])

        # Average over window → 1D per-feature importance
        shap_avg    = np.abs(raw).mean(axis=0)   # (n_features,)
        shap_signed = raw.mean(axis=0)            # (n_features,) signed
        shap_signed = shap_signed.flatten()       # guarantee 1D

        # RUL prediction at this snapshot
        pred_all = self.rul_predictor.predict(df)
        rul_score = float(pred_all[snapshot_idx])

        top_features = self._top_features(
            shap_signed, rul_cols, n=3, invert=False)

        return {
            "shap_values":   shap_signed,
            "feature_names": rul_cols,
            "feature_labels": [self._label(c) for c in rul_cols],
            "top_features":  top_features,
            "rul_score":     rul_score,
        }

    def explain_cnn(self, image: np.ndarray) -> dict:
        """
        CNN Autoencoder reconstruction error heatmap for one snapshot.

        Instead of SHAP (which is expensive and hard to interpret for
        autoencoders with 12,288 input features), we compute the
        per-pixel reconstruction error as a spatial heatmap.

        This tells the RCA agent:
            - WHICH channel (STFT / Mel / GAF) shows highest error
            - WHICH frequency band has the most abnormal patterns
            - Maps to probable fault type via frequency → fault mapping

        Why this works:
            The autoencoder was trained only on healthy bearing spectrograms.
            When it sees a faulty spectrogram, it can't reconstruct the
            abnormal regions well → high per-pixel MSE in those regions.
            Those regions correspond to specific frequency bands where
            bearing fault energy appears (BPFO at ~mid-low, harmonics at mid-high).

        Args:
            image : np.ndarray shape (64, 64, 3) — the spectrogram image
                    from signal_to_image.py for this bearing + snapshot

        Returns dict:
            {
              'heatmap'          : np.ndarray (64, 64, 3) — per-pixel MSE,
              'channel_errors'   : list of 3 floats — mean MSE per channel,
              'worst_channel'    : int — channel index with highest error,
              'worst_channel_name': str — 'STFT spectrogram' / 'Mel ...' / 'GAF',
              'band_errors'      : dict — mean MSE per frequency band,
              'worst_band'       : str — frequency band name with highest error,
              'cv_score'         : float — total reconstruction MSE,
              'cv_flag'          : int — 1 if above threshold, 0 otherwise,
              'fault_from_band'  : str — probable fault type from band mapping,
              'rca_context'      : str — pre-formatted for LLM prompt
            }
        """
        if self.cv_detector is None:
            return {"error": "CNN Autoencoder not provided."}
        if not self.cv_detector.is_fitted_:
            return {"error": "CNN Autoencoder not fitted. Call fit() first."}
        if not TF_AVAILABLE:
            return {"error": "TensorFlow required for CNN explainability."}

        # ── Reconstruct the image ────────────────────────────────────────
        x = image[np.newaxis, ...]                       # (1, 64, 64, 3)
        recon = self.cv_detector.model_.predict(x, verbose=0)  # (1, 64, 64, 3)

        # ── Per-pixel reconstruction error heatmap ───────────────────────
        heatmap = (x[0] - recon[0]) ** 2                 # (64, 64, 3)

        # Total anomaly score (matches cv_anomaly_detector.score_image)
        cv_score = float(np.mean(heatmap))
        cv_flag  = int(cv_score > self.cv_detector.threshold_) \
            if self.cv_detector.threshold_ is not None else 0

        # ── Per-channel analysis ─────────────────────────────────────────
        channel_errors = [float(np.mean(heatmap[:, :, ch]))
                          for ch in range(3)]
        worst_channel  = int(np.argmax(channel_errors))

        # ── Per-frequency-band analysis ──────────────────────────────────
        # Use the worst channel for band decomposition
        worst_ch_heatmap = heatmap[:, :, worst_channel]   # (64, 64)
        band_errors = {}
        for band_name, (row_start, row_end) in CNN_FREQ_BANDS.items():
            band_errors[band_name] = float(
                np.mean(worst_ch_heatmap[row_start:row_end + 1, :]))

        worst_band = max(band_errors, key=band_errors.get)
        fault_from_band = CNN_BAND_FAULT_MAP.get(worst_band, "undetermined")

        # ── Build CNN-specific RCA context ───────────────────────────────
        rca_lines = [
            f"CNN Autoencoder reconstruction error: {cv_score:.6f}"
            f" ({'ANOMALY' if cv_flag else 'NORMAL'})",
            f"Worst channel: {CNN_CHANNEL_NAMES[worst_channel]} "
            f"(MSE: {channel_errors[worst_channel]:.6f})",
            f"Worst frequency band: {worst_band} "
            f"(MSE: {band_errors[worst_band]:.6f})",
            f"Band fault inference: {fault_from_band}",
            "",
            "Per-channel MSE:",
        ]
        for ch_idx, ch_name in enumerate(CNN_CHANNEL_NAMES):
            rca_lines.append(
                f"  - {ch_name}: {channel_errors[ch_idx]:.6f}")
        rca_lines += ["", "Per-band MSE (worst channel):"]
        for band_name, err in sorted(
                band_errors.items(), key=lambda x: x[1], reverse=True):
            rca_lines.append(f"  - {band_name}: {err:.6f}")

        return {
            "heatmap":           heatmap,
            "channel_errors":    channel_errors,
            "worst_channel":     worst_channel,
            "worst_channel_name": CNN_CHANNEL_NAMES[worst_channel],
            "band_errors":       band_errors,
            "worst_band":        worst_band,
            "cv_score":          cv_score,
            "cv_flag":           cv_flag,
            "fault_from_band":   fault_from_band,
            "rca_context":       "\n".join(rca_lines),
        }

    def explain_snapshot(self, df: pd.DataFrame,
                          snapshot_idx: int,
                          image: np.ndarray = None) -> dict:
        """
        Full explanation for one snapshot — all three models combined.

        This is the method the RCA Agent calls.
        Returns a structured context dict ready to inject into an LLM prompt.

        Args:
            df            : full feature matrix DataFrame
            snapshot_idx  : which snapshot to explain
            image         : optional (64,64,3) spectrogram image for CNN explain.
                            If None and cv_detector is set, CNN section is skipped
                            with a note in the context.

        Returns:
            {
              'bearing_id'      : str,
              'snapshot_idx'    : int,
              'anomaly'         : { shap dict from explain_anomaly_score },
              'rul'             : { shap dict from explain_rul },
              'cnn'             : { dict from explain_cnn } or None,
              'probable_fault'  : str — inferred fault type from all evidence,
              'rca_context'     : str — pre-formatted text for LLM prompt
            }
        """
        anomaly_exp = self.explain_anomaly_score(df, snapshot_idx)
        rul_exp     = self.explain_rul(df, snapshot_idx)

        # CNN Autoencoder explanation (optional — needs image input)
        cnn_exp = None
        if self.cv_detector is not None and image is not None:
            cnn_exp = self.explain_cnn(image)

        # Infer probable fault type from ALL available evidence
        all_top = (
            anomaly_exp.get("top_features", []) +
            rul_exp.get("top_features", [])
        )
        probable_fault = self._infer_fault(all_top)

        # If CNN provided a band-based fault inference, use it to corroborate
        if cnn_exp and "fault_from_band" in cnn_exp:
            cnn_fault = cnn_exp["fault_from_band"]
            if probable_fault == "undetermined — review sensor data":
                probable_fault = cnn_fault
            elif cnn_fault != "undetermined":
                probable_fault = (
                    f"{probable_fault} "
                    f"(corroborated by CNN: {cnn_fault})")

        # Build pre-formatted context string for RCA agent LLM prompt
        rca_context = self._build_rca_context(
            anomaly_exp, rul_exp, probable_fault, snapshot_idx,
            cnn_exp=cnn_exp)

        return {
            "bearing_id":    self.bearing_id,
            "snapshot_idx":  snapshot_idx,
            "anomaly":       anomaly_exp,
            "rul":           rul_exp,
            "cnn":           cnn_exp,
            "probable_fault": probable_fault,
            "rca_context":   rca_context,
        }

    # ── Batch explanation ─────────────────────────────────────────────────────

    def explain_anomalies(self, df: pd.DataFrame,
                           anomaly_scores_df: pd.DataFrame,
                           max_snapshots: int = 50) -> pd.DataFrame:
        """
        Explain all flagged anomaly snapshots for this bearing.

        Args:
            df                : full feature matrix
            anomaly_scores_df : output from BearingAnomalyDetector.score_all()
            max_snapshots     : cap to avoid very long runs

        Returns:
            DataFrame with one row per flagged snapshot:
                snapshot_idx, anomaly_score, rul_score,
                top_feature_1, top_feature_2, top_feature_3,
                probable_fault, rca_context
        """
        flag_col    = f"{self.bearing_id}_flag"
        if flag_col not in anomaly_scores_df.columns:
            print(f"No flag column {flag_col} in scores DataFrame.")
            return pd.DataFrame()

        flagged_idx = np.where(
            anomaly_scores_df[flag_col].values == 1)[0][:max_snapshots]

        records = []
        for idx in flagged_idx:
            try:
                exp = self.explain_snapshot(df, int(idx))
                top = exp["anomaly"]["top_features"]
                records.append({
                    "snapshot_idx":  idx,
                    "anomaly_score": exp["anomaly"]["anomaly_score"],
                    "rul_score":     exp["rul"].get("rul_score", None),
                    "top_feature_1": top[0][0] if len(top) > 0 else None,
                    "top_feature_2": top[1][0] if len(top) > 1 else None,
                    "top_feature_3": top[2][0] if len(top) > 2 else None,
                    "probable_fault": exp["probable_fault"],
                    "rca_context":   exp["rca_context"],
                })
            except Exception as e:
                print(f"  Warning: could not explain snapshot {idx} — {e}")

        return pd.DataFrame(records)

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_waterfall(self, df: pd.DataFrame, snapshot_idx: int,
                       output_path: str = None) -> None:
        """
        Waterfall plot for the Isolation Forest at one snapshot.
        Shows each feature pushing the score up (toward anomaly) or down.
        """
        if not SHAP_AVAILABLE:
            return
        exp = self.explain_anomaly_score(df, snapshot_idx)
        sv  = exp["shap_values"]
        labels = [self._label(c) for c in exp["feature_names"]]

        # Sort by absolute value
        order  = np.argsort(np.abs(sv))[::-1][:10]  # top 10
        sv_top = sv[order]
        lb_top = [labels[i] for i in order]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#E24B4A" if v > 0 else "#378ADD" for v in sv_top]
        bars   = ax.barh(lb_top[::-1], sv_top[::-1], color=colors[::-1],
                         height=0.6)
        ax.axvline(0, color="#888888", linewidth=0.8,
                   linestyle="--")
        ax.set_xlabel("SHAP value (contribution to anomaly score)")
        ax.set_title(
            f"SHAP — {self.bearing_id} | snapshot {snapshot_idx}\n"
            f"Anomaly score: {exp['anomaly_score']:.4f}", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Waterfall plot saved → {output_path}")
        plt.close()

    def plot_summary(self, df: pd.DataFrame, n_snapshots: int = 200,
                     output_path: str = None) -> None:
        """
        Summary bar chart — mean absolute SHAP value per feature
        across n_snapshots. Shows which features matter most overall.
        """
        if not SHAP_AVAILABLE:
            return

        cols     = self.feature_names_
        X_raw    = df[cols].values[:n_snapshots]
        X_scaled = self.anomaly_detector.scaler.transform(X_raw)

        shap_vals = self.tree_explainer_.shap_values(X_scaled)  # (N, F)
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        labels    = [self._label(c) for c in cols]

        order     = np.argsort(mean_abs)
        sorted_labels = [labels[i] for i in order]
        sorted_vals   = mean_abs[order]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(sorted_labels, sorted_vals, color="#1D9E75", height=0.6)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(
            f"Feature importance — {self.bearing_id}\n"
            f"(Isolation Forest, first {n_snapshots} snapshots)", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Summary plot saved → {output_path}")
        plt.close()

    def plot_cnn_heatmap(self, image: np.ndarray,
                          snapshot_idx: int = None,
                          output_path: str = None) -> None:
        """
        Plot the CNN reconstruction error heatmap for one spectrogram.

        Shows 4 panels:
            1. Original spectrogram (RGB)
            2. Reconstructed spectrogram
            3. Per-pixel reconstruction error heatmap
            4. Per-channel error bar chart + frequency band breakdown

        Args:
            image        : (64, 64, 3) spectrogram from signal_to_image.py
            snapshot_idx : snapshot number for title
            output_path  : save path (None = show only)
        """
        if self.cv_detector is None or not self.cv_detector.is_fitted_:
            print("CNN detector not available or not fitted.")
            return
        if not TF_AVAILABLE:
            print("TensorFlow required for CNN heatmap plot.")
            return

        cnn_exp = self.explain_cnn(image)
        if "error" in cnn_exp:
            print(f"CNN explain error: {cnn_exp['error']}")
            return

        x     = image[np.newaxis, ...]
        recon = self.cv_detector.model_.predict(x, verbose=0)[0]
        heatmap = cnn_exp["heatmap"]

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        # Panel 1: Original
        axes[0].imshow(image, aspect="auto")
        axes[0].set_title("Original spectrogram", fontsize=9)
        axes[0].axis("off")

        # Panel 2: Reconstruction
        axes[1].imshow(np.clip(recon, 0, 1), aspect="auto")
        axes[1].set_title("Reconstruction", fontsize=9)
        axes[1].axis("off")

        # Panel 3: Error heatmap (mean across channels for visibility)
        error_map = np.mean(heatmap, axis=-1)
        im = axes[2].imshow(error_map, cmap="hot", aspect="auto")
        axes[2].set_title("Reconstruction error", fontsize=9)
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        # Draw frequency band boundaries
        for band_name, (r_start, r_end) in CNN_FREQ_BANDS.items():
            axes[2].axhline(r_start, color="white", linewidth=0.5,
                            linestyle="--", alpha=0.5)

        # Panel 4: Per-band error breakdown
        band_names = list(cnn_exp["band_errors"].keys())
        band_vals  = [cnn_exp["band_errors"][b] for b in band_names]
        colors = ["#E24B4A" if b == cnn_exp["worst_band"]
                  else "#378ADD" for b in band_names]
        axes[3].barh(band_names, band_vals, color=colors, height=0.6)
        axes[3].set_xlabel("Mean MSE", fontsize=8)
        axes[3].set_title("Error by freq band", fontsize=9)
        axes[3].grid(True, axis="x", alpha=0.3)

        snap_str = f" | snapshot {snapshot_idx}" if snapshot_idx else ""
        fig.suptitle(
            f"CNN Autoencoder — {self.bearing_id}{snap_str}\n"
            f"Total MSE: {cnn_exp['cv_score']:.6f} | "
            f"Worst: {cnn_exp['worst_channel_name']} / {cnn_exp['worst_band']}",
            fontsize=10)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"CNN heatmap saved → {output_path}")
        plt.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _label(self, col: str) -> str:
        """Strip bearing prefix from column name and look up human label."""
        # e.g. 'b1_ch1_kurtosis' → 'kurtosis' → 'Kurtosis'
        parts   = col.split("_", 2)
        feature = parts[2] if len(parts) > 2 else col
        return FEATURE_LABELS.get(feature, feature)

    def _top_features(self, shap_vals: np.ndarray, cols: list,
                      n: int = 3, invert: bool = False) -> list:
        """
        Return top-n features by absolute SHAP value.

        Returns list of (human_label, shap_value, direction_str)
        direction: '↑ toward anomaly' or '↓ toward normal'
        """
        shap_arr = np.array(shap_vals)
        order = np.argsort(np.abs(shap_arr))[::-1][:n]
        result = []
        for i in order:
            i         = int(i)   # cast numpy scalar to Python int for list indexing
            val       = float(shap_arr[i])
            label     = self._label(cols[i])
            if invert:
                direction = "↑ toward anomaly" if val > 0 else "↓ toward normal"
            else:
                direction = "↓ reduces RUL" if val < 0 else "↑ increases RUL"
            result.append((label, round(val, 4), direction))
        return result

    def _infer_fault(self, top_features: list) -> str:
        """
        Infer probable fault type from top SHAP features.
        Looks for known fault-frequency features in top contributors.
        """
        for label, val, direction in top_features:
            # Match against raw feature keys
            for key, fault_type in FAULT_INFERENCE_MAP.items():
                if key.replace("_", " ") in label.lower() or \
                   key in label.lower().replace(" ", "_"):
                    return fault_type
        return "undetermined — review sensor data"

    def _build_rca_context(self, anomaly_exp: dict, rul_exp: dict,
                            probable_fault: str,
                            snapshot_idx: int,
                            cnn_exp: dict = None) -> str:
        """
        Build the pre-formatted context string injected into the RCA agent
        LLM prompt. Structured so the LLM can cite specific evidence.

        Includes three sections:
            1. Isolation Forest SHAP — which statistical features drive anomaly
            2. LSTM SHAP — which features drive RUL prediction
            3. CNN Autoencoder heatmap — which spectral regions are abnormal
        """
        top_anomaly = anomaly_exp.get("top_features", [])
        top_rul     = rul_exp.get("top_features", [])
        rul_score   = rul_exp.get("rul_score", None)
        anom_score  = anomaly_exp.get("anomaly_score", None)

        lines = [
            f"Bearing: {self.bearing_id}",
            f"Snapshot index: {snapshot_idx}",
            f"Anomaly score: {anom_score:.4f}" if anom_score else "",
            f"RUL score: {rul_score:.4f} "
            f"({'CRITICAL' if rul_score and rul_score <= 0.15 else 'WARNING' if rul_score and rul_score <= 0.30 else 'NORMAL'})"
            if rul_score else "",
            "",
            "Top anomaly drivers (Isolation Forest SHAP):",
        ]
        for feat, val, direction in top_anomaly:
            lines.append(f"  - {feat}: {val:+.4f} ({direction})")

        lines += ["", "Top RUL drivers (LSTM SHAP):"]
        for feat, val, direction in top_rul:
            lines.append(f"  - {feat}: {val:+.4f} ({direction})")

        # CNN Autoencoder section
        if cnn_exp and "error" not in cnn_exp:
            lines += [
                "",
                "CNN Autoencoder spectral analysis:",
                f"  Reconstruction MSE: {cnn_exp['cv_score']:.6f}"
                f" ({'ANOMALY' if cnn_exp['cv_flag'] else 'NORMAL'})",
                f"  Worst channel: {cnn_exp['worst_channel_name']}",
                f"  Worst frequency band: {cnn_exp['worst_band']}",
                f"  Band fault inference: {cnn_exp['fault_from_band']}",
            ]
        elif self.cv_detector is not None:
            lines += [
                "",
                "CNN Autoencoder: no spectrogram image provided for this snapshot.",
            ]

        lines += [
            "",
            f"Probable fault type: {probable_fault}",
        ]

        return "\n".join(l for l in lines if l is not None)


# ── Multi-Bearing SHAP Explainer ───────────────────────────────────────────────

class BearingShapExplainer:
    """
    Manages one SingleBearingShapExplainer per bearing.

    Same design pattern as BearingAnomalyDetector and BearingRULPredictor —
    consistent interface for the LangGraph orchestrator.

    Args:
        anomaly_detector : fitted BearingAnomalyDetector instance
        rul_predictor    : fitted BearingRULPredictor instance
        cv_detector      : fitted CVAnomalyDetector instance (optional)
    """

    def __init__(self, anomaly_detector, rul_predictor, cv_detector=None):
        self.anomaly_detector  = anomaly_detector
        self.rul_predictor     = rul_predictor
        self.cv_detector       = cv_detector
        self.explainers_: dict[str, SingleBearingShapExplainer] = {}
        self.feature_matrix_: pd.DataFrame | None = None
        self.bearing_ids_: list[str] = []

    def fit(self, feature_matrix_path: str = None,
            df: pd.DataFrame = None,
            n_background: int = 100) -> "BearingShapExplainer":
        """
        Initialise SHAP explainers for all bearings.

        Args:
            feature_matrix_path : path to CSV from features.py (or pass df)
            df                  : pre-loaded feature matrix DataFrame
            n_background        : healthy background sample count
        """
        if df is None:
            if feature_matrix_path is None:
                raise ValueError("Provide feature_matrix_path or df.")
            print(f"Loading feature matrix: {feature_matrix_path}")
            df = pd.read_csv(feature_matrix_path, index_col=0)

        self.feature_matrix_ = df
        self.bearing_ids_    = self.anomaly_detector.bearing_ids_

        print(f"Initialising SHAP explainers for "
              f"{len(self.bearing_ids_)} bearings ...")

        for bid in self.bearing_ids_:
            print(f"  {bid} ...", end=" ")
            exp = SingleBearingShapExplainer(
                bearing_id=bid,
                anomaly_detector=self.anomaly_detector.detectors_[bid],
                rul_predictor=self.rul_predictor.predictors_[bid],
                cv_detector=self.cv_detector,
            )
            exp.fit(df, n_background=n_background)
            self.explainers_[bid] = exp
            print("✓")

        print(f"\n✓ All {len(self.bearing_ids_)} SHAP explainers ready.")
        return self

    # ── Agent-facing interface ────────────────────────────────────────────────

    def explain_snapshot(self, snapshot_index: int,
                          images: dict = None) -> dict:
        """
        Full SHAP explanation for all bearings at one snapshot.
        This is the primary method called by the Orchestrator Agent.

        Args:
            snapshot_index : which snapshot to explain
            images         : optional dict mapping bearing_id → (64,64,3) image
                             e.g. {'b1_ch1': np.ndarray, 'b2_ch1': ...}
                             If provided, CNN heatmap analysis is included.

        Returns:
            {
              'b1_ch1': { bearing explanation dict },
              'b2_ch1': { ... },
              ...
              'most_anomalous_bearing': str,
              'system_rca_context': str  ← inject directly into LLM prompt
            }
        """
        if not self.explainers_:
            raise RuntimeError("Call fit() first.")

        results         = {}
        worst_score     = -np.inf
        worst_bearing   = None

        for bid, exp in self.explainers_.items():
            img = images.get(bid) if images else None
            explanation    = exp.explain_snapshot(
                self.feature_matrix_, snapshot_index, image=img)
            results[bid]   = explanation

            anom_score = explanation["anomaly"].get("anomaly_score", 0)
            if anom_score > worst_score:
                worst_score   = anom_score
                worst_bearing = bid

        results["most_anomalous_bearing"] = worst_bearing

        # Combined context for the RCA agent — worst bearing leads
        if worst_bearing:
            results["system_rca_context"] = (
                f"=== System SHAP Explanation — Snapshot {snapshot_index} ===\n\n"
                + results[worst_bearing]["rca_context"]
            )

        return results

    def explain_all_anomalies(self, anomaly_scores_df: pd.DataFrame,
                               output_dir: str = None) -> pd.DataFrame:
        """
        Explain every flagged anomaly across all bearings.

        Args:
            anomaly_scores_df : output from BearingAnomalyDetector.score_all()
            output_dir        : if provided, saves per-bearing CSV reports

        Returns:
            Combined DataFrame with bearing_id column added
        """
        all_records = []

        for bid, exp in self.explainers_.items():
            print(f"  Explaining anomalies for {bid} ...")
            df_exp = exp.explain_anomalies(
                self.feature_matrix_, anomaly_scores_df)
            if not df_exp.empty:
                df_exp.insert(0, "bearing_id", bid)
                all_records.append(df_exp)

                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    out = os.path.join(output_dir, f"shap_{bid}.csv")
                    df_exp.to_csv(out, index=False)
                    print(f"    Saved → {out}")

        if not all_records:
            print("No anomalies to explain.")
            return pd.DataFrame()

        combined = pd.concat(all_records, ignore_index=True)
        print(f"\n✓ Explained {len(combined)} anomalous snapshots total.")
        return combined

    def plot_all_summaries(self, output_dir: str) -> None:
        """Generate SHAP summary bar chart for each bearing."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for bid, exp in self.explainers_.items():
            exp.plot_summary(
                self.feature_matrix_,
                output_path=os.path.join(output_dir,
                                         f"shap_summary_{bid}.png"))
            print(f"  Summary plot → {output_dir}/shap_summary_{bid}.png")

    def plot_waterfall_for_snapshot(self, snapshot_index: int,
                                     output_dir: str) -> None:
        """Generate waterfall SHAP plot for each bearing at one snapshot."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for bid, exp in self.explainers_.items():
            exp.plot_waterfall(
                self.feature_matrix_,
                snapshot_index,
                output_path=os.path.join(
                    output_dir,
                    f"shap_waterfall_{bid}_snap{snapshot_index}.png"))

    def plot_cnn_heatmaps(self, images: dict, snapshot_index: int,
                           output_dir: str) -> None:
        """
        Generate CNN reconstruction error heatmap for each bearing.

        Args:
            images          : dict mapping bearing_id → (64,64,3) image
            snapshot_index  : snapshot number for titles
            output_dir      : where to save plots
        """
        if self.cv_detector is None:
            print("CNN detector not provided — skipping heatmap plots.")
            return

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for bid, exp in self.explainers_.items():
            img = images.get(bid)
            if img is None:
                print(f"  No image for {bid} — skipping.")
                continue
            exp.plot_cnn_heatmap(
                img, snapshot_idx=snapshot_index,
                output_path=os.path.join(
                    output_dir,
                    f"cnn_heatmap_{bid}_snap{snapshot_index}.png"))
        print(f"  CNN heatmap plots saved → {output_dir}/")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # These imports must be explicit so pickle can locate the classes
    # when deserializing saved model files
    from isolation_forest import BearingAnomalyDetector, SingleBearingDetector
    from rul_lstm import BearingRULPredictor, SingleBearingRUL, LSTMRULModel

    if len(sys.argv) < 4:
        print("Usage: python shap_explainer.py <feature_matrix.csv> "
              "<models_dir> <output_dir> [snapshot_index] [images_dir]")
        print("  feature_matrix.csv : output from features.py")
        print("  models_dir         : directory with saved IF + LSTM + CV models")
        print("  output_dir         : where to save SHAP plots and reports")
        print("  snapshot_index     : single snapshot to explain (default: last)")
        print("  images_dir         : path to images/all/ from signal_to_image.py")
        print("                       (optional — CNN heatmaps skipped if missing)")
        sys.exit(1)

    feature_csv   = sys.argv[1]
    models_dir    = sys.argv[2]
    output_dir    = sys.argv[3]

    df = pd.read_csv(feature_csv, index_col=0)
    snapshot_idx = int(sys.argv[4]) if len(sys.argv) > 4 else len(df) - 1
    images_dir   = sys.argv[5] if len(sys.argv) > 5 else None

    # ── Paths for each model ─────────────────────────────────────────────
    if_models_dir  = os.path.join(models_dir, "if", "models")
    rul_models_dir = os.path.join(models_dir, "rul", "models")
    cv_model_dir   = os.path.join(models_dir, "cv", "model")

    if_exists  = os.path.isdir(if_models_dir) and any(Path(if_models_dir).glob("detector_*.pkl"))
    rul_exists = os.path.isdir(rul_models_dir) and any(Path(rul_models_dir).glob("rul_*.pt"))
    cv_exists  = os.path.isdir(cv_model_dir) and os.path.isfile(
        os.path.join(cv_model_dir, "autoencoder.keras"))

    # ── Load Isolation Forest ────────────────────────────────────────────
    print("Loading Isolation Forest models ...")
    anomaly_det = BearingAnomalyDetector()
    anomaly_det.fit_from_df(df)
    if if_exists:
        anomaly_det.load_models(if_models_dir)
        print(f"  Loaded saved models from {if_models_dir}")
    else:
        print(f"  Training IF models (first run) ...")
        anomaly_det.save_models(if_models_dir)
    scores_df = anomaly_det.score_all()

    # ── Load LSTM RUL ────────────────────────────────────────────────────
    print("Loading LSTM RUL models ...")
    rul_pred = BearingRULPredictor()
    if rul_exists:
        rul_pred.feature_matrix_ = df
        rul_pred.load_models(rul_models_dir)
        rul_pred.bearing_ids_ = sorted(rul_pred.predictors_.keys())
        print(f"  Loaded saved models from {rul_models_dir}")
    else:
        print(f"  Training LSTM models (first run) ...")
        rul_pred.fit_from_df(df)
        rul_pred.save_models(rul_models_dir)

    # ── Load CNN Autoencoder (graceful fallback) ─────────────────────────
    cv_detector = None
    if cv_exists:
        try:
            from cv_anomaly_detector import CVAnomalyDetector
            cv_detector = CVAnomalyDetector()
            cv_detector.load(cv_model_dir)
            print(f"  CNN Autoencoder loaded from {cv_model_dir}")
        except Exception as e:
            print(f"  CNN Autoencoder skipped — {e}")
            cv_detector = None
    else:
        print("  CNN Autoencoder not found — skipping CNN explainability.")
        print("  (Run signal_to_image.py → cv_anomaly_detector.py first)")

    # ── Load spectrogram images for the target snapshot (graceful) ───────
    snapshot_images = None
    if cv_detector is not None and images_dir is not None:
        # Images are named: {snapshot_filename}_{bearing_id}.npy
        # We need to find all images matching the target snapshot index
        images_path = Path(images_dir)
        if images_path.is_dir():
            all_npy = sorted(images_path.glob("*.npy"))
            if all_npy:
                # Snapshot files are sorted chronologically — index directly
                # Group by snapshot: every 4 consecutive files = 1 snapshot × 4 bearings
                bearing_ids = anomaly_det.bearing_ids_
                n_bearings  = len(bearing_ids)

                # Each snapshot produces n_bearings images, sorted together
                start_file_idx = snapshot_idx * n_bearings
                end_file_idx   = start_file_idx + n_bearings

                if end_file_idx <= len(all_npy):
                    snapshot_images = {}
                    snapshot_files  = all_npy[start_file_idx:end_file_idx]
                    for fpath in snapshot_files:
                        # Parse bearing_id from filename: ..._{bid}.npy
                        stem   = fpath.stem
                        parts  = stem.rsplit("_", 2)
                        bid    = "_".join(parts[1:]) if len(parts) >= 3 else None
                        if bid in bearing_ids:
                            snapshot_images[bid] = np.load(str(fpath))

                    if snapshot_images:
                        print(f"  Loaded {len(snapshot_images)} spectrogram images "
                              f"for snapshot {snapshot_idx}")
                    else:
                        print(f"  Could not match image files to bearings — "
                              f"CNN heatmaps skipped.")
                        snapshot_images = None
                else:
                    print(f"  Snapshot {snapshot_idx} out of range for images — "
                          f"CNN heatmaps skipped.")
            else:
                print(f"  No .npy files in {images_dir} — CNN heatmaps skipped.")
        else:
            print(f"  Images directory not found: {images_dir} — "
                  f"CNN heatmaps skipped.")
    elif cv_detector is not None and images_dir is None:
        print("  No images_dir provided — CNN heatmaps skipped for single snapshot.")
        print("  (Pass images/all/ as 5th argument to enable CNN heatmaps)")

    # ── Build SHAP explainers ────────────────────────────────────────────
    explainer = BearingShapExplainer(anomaly_det, rul_pred,
                                     cv_detector=cv_detector)
    explainer.fit(df=df, n_background=100)

    # ── Explain single snapshot ──────────────────────────────────────────
    print(f"\nExplaining snapshot {snapshot_idx} ...")
    result = explainer.explain_snapshot(snapshot_idx, images=snapshot_images)

    print("\n── RCA Context (inject into LLM prompt) ──")
    print(result.get("system_rca_context", "No context generated."))

    # ── Batch explain all anomalies ──────────────────────────────────────
    print("\nExplaining all anomalies ...")
    report = explainer.explain_all_anomalies(
        scores_df, output_dir=output_dir)
    if not report.empty:
        report.to_csv(
            os.path.join(output_dir, "shap_report.csv"), index=False)
        print(f"Full report saved → {output_dir}/shap_report.csv")

    # ── Plots ────────────────────────────────────────────────────────────
    explainer.plot_all_summaries(output_dir)
    explainer.plot_waterfall_for_snapshot(snapshot_idx, output_dir)

    # CNN heatmap plots (only if both model and images are available)
    if cv_detector is not None and snapshot_images is not None:
        explainer.plot_cnn_heatmaps(
            snapshot_images, snapshot_idx, output_dir)

    print(f"\n✓ All SHAP outputs saved → {output_dir}")