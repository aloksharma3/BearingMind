"""
rul_lstm.py — LSTM Remaining Useful Life Predictor
Industrial AI Predictive Maintenance | BearingMind

Predicts how much useful life remains in a bearing using a sliding-window
LSTM trained on the 64-feature matrix produced by features.py.

RUL Label Strategy (no ground-truth labels in NASA IMS):
    - Snapshot 1   → RUL = 1.0  (100% life remaining)
    - Last snapshot → RUL = 0.0  (failure)
    - Everything between → linear decay from 1.0 → 0.0
    This is the standard approach used in all published IMS benchmark papers.

Architecture:
    Input:  (window_size=30, n_features=16) per bearing
    LSTM 1: 128 units, return sequences=True, dropout 0.2
    LSTM 2: 64 units, return sequences=False, dropout 0.2
    Dense:  32 units, ReLU
    Output: 1 unit, sigmoid → RUL score in [0, 1]

Pipeline position:
    features.py → [feature_matrix.csv] → rul_lstm.py
                                        → [rul_predictions.csv]
                                        → [rul_plot.png]

Usage:
    from src.rul_lstm import BearingRULPredictor

    predictor = BearingRULPredictor(window_size=30, epochs=50)
    predictor.fit("results/feature_matrix.csv")
    preds = predictor.predict_all()
    preds.to_csv("results/rul_predictions.csv")
    predictor.plot(preds, "results/rul_plot.png")
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── PyTorch import ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install torch")
    # Stub base classes so class definitions don't crash at import time
    class Dataset:
        pass
    class DataLoader:
        pass
    class nn:
        class Module:
            pass
        class LSTM:
            pass
        class Linear:
            pass
        class Dropout:
            pass
        class ReLU:
            pass
        class Sigmoid:
            pass
        class MSELoss:
            pass
        @staticmethod
        def utils():
            pass

from sklearn.preprocessing import StandardScaler

# ── Constants ──────────────────────────────────────────────────────────────────

WINDOW_SIZE     = 30     # look-back window: 30 snapshots ≈ 5 hours on IMS rig
HIDDEN_SIZE     = 128    # LSTM hidden units (layer 1)
HIDDEN_SIZE_2   = 64     # LSTM hidden units (layer 2)
DENSE_SIZE      = 32     # dense layer before output
DROPOUT         = 0.2
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 32
EPOCHS_DEFAULT  = 50
RANDOM_SEED     = 42

# RUL alert thresholds — what the agent uses to decide urgency
RUL_CRITICAL    = 0.15   # below 15% life → critical alert
RUL_WARNING     = 0.30   # below 30% life → warning


# ── RUL Label Generator ────────────────────────────────────────────────────────

def make_rul_labels(n_snapshots: int) -> np.ndarray:
    """
    Linear RUL labels from 1.0 (healthy) → 0.0 (failure).

    This is the standard approach for run-to-failure datasets with no
    explicit failure timestamps — used in all published IMS benchmark papers.

    Args:
        n_snapshots : total number of snapshots in the dataset

    Returns:
        np.ndarray of shape (n_snapshots,) with values in [0, 1]

    Example:
        984 snapshots → [1.0, 0.999, 0.998, ..., 0.001, 0.0]
    """
    return np.linspace(1.0, 0.0, n_snapshots)


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class BearingWindowDataset(Dataset):
    """
    Sliding window dataset for LSTM training.

    Each sample is:
        X : (window_size, n_features) — last W snapshots of features
        y : scalar RUL label at the END of the window

    Why sliding windows?
        The LSTM needs temporal context — not just "what are the features
        right now" but "how have the features been changing over the last
        W snapshots". The window gives it that history.

    Args:
        features  : np.ndarray shape (n_snapshots, n_features)
        labels    : np.ndarray shape (n_snapshots,) — RUL values
        window    : number of past snapshots to include
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 window: int = WINDOW_SIZE):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels,   dtype=torch.float32)
        self.window = window

    def __len__(self) -> int:
        # First valid index is window_size (need W past snapshots)
        return len(self.X) - self.window

    def __getitem__(self, idx: int):
        x_window = self.X[idx : idx + self.window]          # (W, F)
        y_label  = self.y[idx + self.window]                 # scalar
        return x_window, y_label


# ── LSTM Model ─────────────────────────────────────────────────────────────────

class LSTMRULModel(nn.Module):
    """
    Two-layer stacked LSTM for RUL regression.

    Input  : (batch, window_size, n_features)
    Output : (batch, 1) — predicted RUL in [0, 1]

    Why two LSTM layers?
        Layer 1 captures short-term vibration patterns (spikes, transients).
        Layer 2 captures longer-term degradation trends across the window.
        Stacking gives the model richer temporal representation.

    Why sigmoid output?
        RUL is bounded [0, 1] — sigmoid naturally enforces this constraint
        without needing to clip predictions post-hoc.
    """

    def __init__(self, n_features: int, hidden_size: int = HIDDEN_SIZE,
                 hidden_size_2: int = HIDDEN_SIZE_2, dense_size: int = DENSE_SIZE,
                 dropout: float = DROPOUT):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.drop1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.drop2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_size_2, dense_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features)
        out, _ = self.lstm1(x)          # (batch, window, hidden1)
        out = self.drop1(out)

        out, _ = self.lstm2(out)        # (batch, window, hidden2)
        out = self.drop2(out)

        out = out[:, -1, :]             # take last timestep: (batch, hidden2)
        out = self.relu(self.fc1(out))  # (batch, dense)
        out = self.sigmoid(self.fc2(out))  # (batch, 1)
        return out                      # (batch, 1) — required by SHAP GradientExplainer


# ── Per-Bearing RUL Predictor ──────────────────────────────────────────────────

class SingleBearingRUL:
    """
    LSTM RUL predictor for one bearing.

    Mirrors the design of SingleBearingDetector in isolation_forest.py —
    one model per bearing, same interface, same fit/predict pattern.

    Args:
        bearing_id  : e.g. 'b1_ch1'
        window_size : sliding window length
        hidden_size : LSTM layer 1 units
        epochs      : max training epochs
        batch_size  : training batch size
    """

    def __init__(self, bearing_id: str, window_size: int = WINDOW_SIZE,
                 hidden_size: int = HIDDEN_SIZE, epochs: int = EPOCHS_DEFAULT,
                 batch_size: int = BATCH_SIZE):
        self.bearing_id  = bearing_id
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.scaler_     = StandardScaler()
        self.model_      = None
        self.feature_cols_ = None
        self.train_loss_   = []
        self.val_loss_     = []
        self.is_fitted_    = False
        self.device_       = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def _select_features(self, df: pd.DataFrame) -> np.ndarray:
        """Select this bearing's feature columns from the full matrix."""
        cols = [c for c in df.columns if c.startswith(self.bearing_id)]
        self.feature_cols_ = cols
        return df[cols].values

    def fit(self, df: pd.DataFrame,
            val_split: float = 0.15) -> "SingleBearingRUL":
        """
        Train LSTM on the full feature matrix for this bearing.

        RUL labels are auto-generated (linear 1→0 over all snapshots).
        Last val_split fraction is used for validation — this preserves
        temporal order (no random shuffling — critical for time-series).

        Args:
            df        : full feature matrix DataFrame (all snapshots)
            val_split : fraction of snapshots held out for validation
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")

        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        X_raw = self._select_features(df)
        labels = make_rul_labels(len(X_raw))

        # Scale features — fit on training portion only (no data leakage)
        n_val   = max(1, int(len(X_raw) * val_split))
        n_train = len(X_raw) - n_val

        X_scaled = np.zeros_like(X_raw, dtype=np.float32)
        X_scaled[:n_train] = self.scaler_.fit_transform(X_raw[:n_train])
        X_scaled[n_train:] = self.scaler_.transform(X_raw[n_train:])

        # Datasets — preserve temporal order
        train_ds = BearingWindowDataset(
            X_scaled[:n_train], labels[:n_train], self.window_size)
        val_ds   = BearingWindowDataset(
            X_scaled[n_train:], labels[n_train:], self.window_size)

        # Handle edge case: validation set too small for a window
        if len(val_ds) < 1:
            val_ds = train_ds

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        # Build model
        n_features   = X_scaled.shape[1]
        self.model_  = LSTMRULModel(n_features, self.hidden_size).to(self.device_)
        optimizer    = torch.optim.Adam(self.model_.parameters(), lr=LEARNING_RATE)
        scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6)
        criterion    = nn.MSELoss()

        best_val     = float("inf")
        best_weights = None
        patience     = 10
        patience_ctr = 0

        for epoch in range(self.epochs):
            # Training
            self.model_.train()
            train_losses = []
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                optimizer.zero_grad()
                pred = self.model_(xb).squeeze(1)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model_.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self.device_), yb.to(self.device_)
                    pred = self.model_(xb).squeeze(1)
                    val_losses.append(criterion(pred, yb).item())

            t_loss = float(np.mean(train_losses))
            v_loss = float(np.mean(val_losses))
            self.train_loss_.append(t_loss)
            self.val_loss_.append(v_loss)
            scheduler.step(v_loss)

            if v_loss < best_val:
                best_val     = v_loss
                best_weights = {k: v.cpu().clone()
                                for k, v in self.model_.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}/{self.epochs} | "
                      f"train={t_loss:.4f} val={v_loss:.4f}")

            if patience_ctr >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        # Restore best weights
        if best_weights:
            self.model_.load_state_dict(
                {k: v.to(self.device_) for k, v in best_weights.items()})

        self.is_fitted_ = True
        print(f"  ✓ {self.bearing_id}: best val_loss={best_val:.4f}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict RUL score for every snapshot in df.

        Returns array of length (n_snapshots - window_size).
        First window_size snapshots have no prediction (not enough history).
        We pad those with 1.0 (healthy assumption at start).

        Returns:
            np.ndarray shape (n_snapshots,) with RUL scores in [0, 1]
        """
        if not self.is_fitted_:
            raise RuntimeError(f"{self.bearing_id}: call fit() first.")

        X_raw    = df[self.feature_cols_].values
        X_scaled = self.scaler_.transform(X_raw).astype(np.float32)
        labels   = make_rul_labels(len(X_raw))

        ds = BearingWindowDataset(X_scaled, labels, self.window_size)
        dl = DataLoader(ds, batch_size=128, shuffle=False)

        self.model_.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in dl:
                xb   = xb.to(self.device_)
                pred = self.model_(xb).squeeze(1)
                preds.extend(pred.cpu().numpy().tolist())

        # Pad beginning with 1.0 (no prediction for first window_size snapshots)
        padded = np.ones(len(X_raw), dtype=np.float32)
        padded[self.window_size:] = np.array(preds)
        return padded

    def alert_level(self, rul_score: float) -> str:
        """
        Convert RUL score to human-readable alert level.
        Used by the Alert Agent downstream.

        Returns: 'CRITICAL', 'WARNING', or 'NORMAL'
        """
        if rul_score <= RUL_CRITICAL:
            return "CRITICAL"
        elif rul_score <= RUL_WARNING:
            return "WARNING"
        return "NORMAL"

    def save(self, path: str) -> None:
        """Save model weights, scaler, and metadata."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":   self.model_.state_dict(),
            "scaler":        self.scaler_,
            "feature_cols":  self.feature_cols_,
            "window_size":   self.window_size,
            "hidden_size":   self.hidden_size,
            "n_features":    len(self.feature_cols_),
            "train_loss":    self.train_loss_,
            "val_loss":      self.val_loss_,
        }, path)

    @classmethod
    def load(cls, path: str, bearing_id: str) -> "SingleBearingRUL":
        """Load a saved model."""
        # weights_only=False needed because checkpoint contains StandardScaler
        # alongside model weights. Safe here since we only load our own saved files.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(bearing_id=bearing_id,
                  window_size=checkpoint["window_size"],
                  hidden_size=checkpoint["hidden_size"])
        obj.scaler_       = checkpoint["scaler"]
        obj.feature_cols_ = checkpoint["feature_cols"]
        obj.train_loss_   = checkpoint["train_loss"]
        obj.val_loss_     = checkpoint["val_loss"]
        n_features        = checkpoint["n_features"]
        obj.model_        = LSTMRULModel(n_features, obj.hidden_size).to(obj.device_)
        obj.model_.load_state_dict(checkpoint["model_state"])
        obj.is_fitted_    = True
        return obj


# ── Multi-Bearing RUL Predictor ────────────────────────────────────────────────

class BearingRULPredictor:
    """
    Manages one SingleBearingRUL per bearing.

    Same design pattern as BearingAnomalyDetector — consistent interface
    makes it easy to wire both into the LangGraph orchestrator later.

    Args:
        window_size : sliding window length (default 30)
        hidden_size : LSTM hidden units (default 128)
        epochs      : max training epochs (default 50)
        batch_size  : training batch size (default 32)
    """

    def __init__(self, window_size: int = WINDOW_SIZE,
                 hidden_size: int = HIDDEN_SIZE,
                 epochs: int = EPOCHS_DEFAULT,
                 batch_size: int = BATCH_SIZE):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.predictors_: dict[str, SingleBearingRUL] = {}
        self.feature_matrix_: pd.DataFrame | None = None
        self.bearing_ids_: list[str] = []

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, feature_matrix_path: str) -> "BearingRULPredictor":
        """
        Load feature matrix CSV and fit one LSTM per bearing.

        Args:
            feature_matrix_path : path to CSV produced by features.py
        """
        print(f"Loading feature matrix: {feature_matrix_path}")
        df = pd.read_csv(feature_matrix_path, index_col=0)
        return self.fit_from_df(df)

    def fit_from_df(self, df: pd.DataFrame) -> "BearingRULPredictor":
        """Fit from an already-loaded feature matrix DataFrame."""
        self.feature_matrix_ = df

        # Infer bearing IDs from column prefixes — same logic as isolation_forest.py
        self.bearing_ids_ = sorted(set(
            "_".join(c.split("_")[:2]) for c in df.columns
        ))

        print(f"Fitting RUL predictors for {len(self.bearing_ids_)} bearings "
              f"({len(df)} snapshots each)")
        print(f"Window size: {self.window_size} | "
              f"Epochs: {self.epochs} | Device: "
              f"{'cuda' if torch.cuda.is_available() else 'cpu'}\n")

        for bid in self.bearing_ids_:
            print(f"── {bid} ──")
            predictor = SingleBearingRUL(
                bearing_id=bid,
                window_size=self.window_size,
                hidden_size=self.hidden_size,
                epochs=self.epochs,
                batch_size=self.batch_size,
            )
            predictor.fit(df)
            self.predictors_[bid] = predictor

        print(f"\n✓ All {len(self.bearing_ids_)} LSTM models fitted.")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_all(self) -> pd.DataFrame:
        """
        Predict RUL for every snapshot for every bearing.

        Returns:
            DataFrame with columns:
                b1_ch1_rul, b1_ch1_alert,
                b2_ch1_rul, b2_ch1_alert, ...
                min_rul, system_alert
            Index = snapshot filename (same as feature_matrix_)
        """
        if not self.predictors_:
            raise RuntimeError("Call fit() before predict_all().")

        results = {}
        for bid, predictor in self.predictors_.items():
            rul_scores = predictor.predict(self.feature_matrix_)
            alerts     = [predictor.alert_level(s) for s in rul_scores]
            results[f"{bid}_rul"]   = rul_scores
            results[f"{bid}_alert"] = alerts

        df_out = pd.DataFrame(results, index=self.feature_matrix_.index)

        # System-level: most critical bearing drives the system alert
        rul_cols = [c for c in df_out.columns if c.endswith("_rul")]
        df_out["min_rul"] = df_out[rul_cols].min(axis=1)
        df_out["system_alert"] = df_out["min_rul"].apply(
            lambda s: "CRITICAL" if s <= RUL_CRITICAL
                      else ("WARNING" if s <= RUL_WARNING else "NORMAL")
        )

        # Summary
        print("\nRUL prediction summary:")
        for bid in self.bearing_ids_:
            final_rul   = df_out[f"{bid}_rul"].iloc[-1]
            min_rul     = df_out[f"{bid}_rul"].min()
            alert       = df_out[f"{bid}_alert"].iloc[-1]
            print(f"  {bid}: final RUL={final_rul:.3f} | "
                  f"min RUL={min_rul:.3f} | last alert={alert}")

        return df_out

    def predict_snapshot(self, snapshot_index: int) -> dict:
        """
        Predict RUL for a single snapshot index.
        Used by the orchestrator agent for real-time scoring.

        Returns dict ready for the RCA agent:
            {
              'b1_ch1': {'rul': 0.12, 'alert': 'CRITICAL', 'days_estimate': 3},
              ...
              'system_alert': 'CRITICAL',
              'most_critical_bearing': 'b1_ch1'
            }
        """
        if self.feature_matrix_ is None:
            raise RuntimeError("No feature matrix loaded. Call fit() first.")

        result     = {}
        worst_rul  = 1.0
        worst_bid  = None

        for bid, predictor in self.predictors_.items():
            # Get a window ending at snapshot_index
            start = max(0, snapshot_index - self.window_size)
            df_window = self.feature_matrix_.iloc[start:snapshot_index + 1]
            preds = predictor.predict(df_window)
            rul   = float(preds[-1])

            # Rough days estimate: IMS 2nd test ran ~7 days total (984 snapshots)
            # Each snapshot ≈ 10 minutes → 984 snapshots ≈ 164 hours
            total_hours    = 164.0
            days_remaining = round(rul * total_hours / 24, 1)

            result[bid] = {
                "rul":           round(rul, 4),
                "alert":         predictor.alert_level(rul),
                "days_estimate": days_remaining,
            }

            if rul < worst_rul:
                worst_rul = rul
                worst_bid = bid

        result["system_alert"]           = (
            "CRITICAL" if worst_rul <= RUL_CRITICAL
            else ("WARNING" if worst_rul <= RUL_WARNING else "NORMAL")
        )
        result["most_critical_bearing"]  = worst_bid
        result["system_rul"]             = round(worst_rul, 4)
        return result

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare predicted RUL vs true linear RUL labels.

        Returns per-bearing MAE and RMSE — useful for the README results table.
        """
        n = len(self.feature_matrix_)
        true_rul = make_rul_labels(n)
        metrics  = {}

        for bid in self.bearing_ids_:
            pred = predictions_df[f"{bid}_rul"].values
            mae  = float(np.mean(np.abs(pred - true_rul)))
            rmse = float(np.sqrt(np.mean((pred - true_rul) ** 2)))
            metrics[bid] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4)}

        df_metrics = pd.DataFrame(metrics).T
        print("\nEvaluation (vs linear RUL labels):")
        print(df_metrics.to_string())
        return df_metrics

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot(self, predictions_df: pd.DataFrame,
             output_path: str = None, show: bool = False) -> None:
        """
        Plot predicted RUL vs true RUL for each bearing.
        Critical and warning threshold lines shown for context.
        """
        n          = len(self.feature_matrix_)
        true_rul   = make_rul_labels(n)
        x          = np.arange(n)
        n_bearings = len(self.bearing_ids_)

        fig, axes = plt.subplots(n_bearings, 1,
                                 figsize=(14, 3.5 * n_bearings), sharex=True)
        if n_bearings == 1:
            axes = [axes]

        for ax, bid in zip(axes, self.bearing_ids_):
            pred = predictions_df[f"{bid}_rul"].values

            ax.plot(x, true_rul, color="gray",      linewidth=1.0,
                    linestyle="--", alpha=0.6, label="True RUL (linear)")
            ax.plot(x, pred,     color="steelblue", linewidth=0.9,
                    alpha=0.9, label="Predicted RUL")

            ax.axhline(RUL_WARNING,  color="orange", linewidth=1.0,
                       linestyle=":", label=f"Warning ({RUL_WARNING})")
            ax.axhline(RUL_CRITICAL, color="red",    linewidth=1.0,
                       linestyle=":", label=f"Critical ({RUL_CRITICAL})")

            # Shade critical zone
            ax.axhspan(0, RUL_CRITICAL, color="red",    alpha=0.06)
            ax.axhspan(RUL_CRITICAL, RUL_WARNING,
                       color="orange", alpha=0.06)

            ax.set_ylabel("RUL score", fontsize=9)
            ax.set_title(f"{bid.upper()} — LSTM RUL Prediction", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Snapshot index (time →)", fontsize=9)
        fig.suptitle(
            "LSTM Remaining Useful Life Prediction — NASA IMS Bearing Dataset",
            fontsize=12, y=1.01)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved → {output_path}")
        if show:
            plt.show()
        plt.close()

    def plot_training_curves(self, output_path: str = None) -> None:
        """Plot train/val loss curves for each bearing's LSTM."""
        n_bearings = len(self.bearing_ids_)
        fig, axes  = plt.subplots(1, n_bearings,
                                  figsize=(5 * n_bearings, 4), sharey=True)
        if n_bearings == 1:
            axes = [axes]

        for ax, bid in zip(axes, self.bearing_ids_):
            p = self.predictors_[bid]
            ax.plot(p.train_loss_, label="Train", color="steelblue")
            ax.plot(p.val_loss_,   label="Val",   color="orange")
            ax.set_title(f"{bid}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("LSTM Training Curves — RUL Predictor", fontsize=12)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Training curves saved → {output_path}")
        plt.close()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_models(self, directory: str) -> None:
        """Save all per-bearing LSTM models."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        for bid, predictor in self.predictors_.items():
            path = os.path.join(directory, f"rul_{bid}.pt")
            predictor.save(path)
        print(f"✓ RUL models saved → {directory}")

    def load_models(self, directory: str) -> None:
        """Load previously saved LSTM models."""
        for path in Path(directory).glob("rul_*.pt"):
            bid = path.stem.replace("rul_", "")
            predictor = SingleBearingRUL.load(str(path), bearing_id=bid)
            self.predictors_[bid] = predictor
            self.bearing_ids_.append(bid)
        self.bearing_ids_ = sorted(self.bearing_ids_)
        print(f"✓ Loaded {len(self.predictors_)} RUL models from {directory}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python rul_lstm.py <feature_matrix.csv> "
              "<output_dir> [epochs] [window_size]")
        print("  feature_matrix.csv : output from features.py")
        print("  output_dir         : where to save predictions, plots, models")
        print("  epochs             : training epochs (default 50)")
        print("  window_size        : look-back window (default 30)")
        sys.exit(1)

    feature_csv = sys.argv[1]
    output_dir  = sys.argv[2]
    epochs      = int(sys.argv[3]) if len(sys.argv) > 3 else EPOCHS_DEFAULT
    window      = int(sys.argv[4]) if len(sys.argv) > 4 else WINDOW_SIZE

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    predictor = BearingRULPredictor(window_size=window, epochs=epochs)
    predictor.fit(feature_csv)

    preds = predictor.predict_all()
    preds.to_csv(os.path.join(output_dir, "rul_predictions.csv"))
    print(f"\nPredictions saved → {output_dir}/rul_predictions.csv")

    metrics = predictor.evaluate(preds)
    metrics.to_csv(os.path.join(output_dir, "rul_metrics.csv"))

    predictor.plot(preds,
                   output_path=os.path.join(output_dir, "rul_plot.png"))
    predictor.plot_training_curves(
        output_path=os.path.join(output_dir, "rul_training_curves.png"))

    predictor.save_models(os.path.join(output_dir, "models"))
    print(f"\n✓ All outputs saved → {output_dir}")