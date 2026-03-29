"""
isolation_forest.py — Isolation Forest Anomaly Detector
Industrial AI Predictive Maintenance

Trains an unsupervised Isolation Forest on the first N_NORMAL snapshots
(healthy bearing data), then scores every snapshot to produce a
time-series anomaly signal for each bearing.

Design philosophy (matches what Siemens/ABB ship):
  - Train on NORMAL data only — no failure labels needed
  - Per-bearing models — each bearing gets its own detector
  - Outputs both raw scores and binary flags
  - Threshold set from training distribution (99th percentile)

Pipeline position:
    features.py → [feature_matrix.csv] → isolation_forest.py
                                        → [anomaly_scores.csv]
                                        → [anomaly_flags.csv]

Usage:
    from src.isolation_forest import BearingAnomalyDetector

    detector = BearingAnomalyDetector(n_normal=500)
    detector.fit("results/feature_matrix.csv")
    scores = detector.score_all()
    scores.to_csv("results/anomaly_scores.csv")
    detector.plot("results/anomaly_plot.png")
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

N_NORMAL_DEFAULT    = 500    # first N snapshots treated as healthy
CONTAMINATION       = 0.01   # expected fraction of anomalies in training data
N_ESTIMATORS        = 200    # number of isolation trees
RANDOM_STATE        = 42
ANOMALY_PERCENTILE  = 99     # threshold: flag top 1% of scores as anomalies


# ── Per-Bearing Detector ──────────────────────────────────────────────────────

class SingleBearingDetector:
    """
    Isolation Forest + StandardScaler for one bearing.

    Isolation Forest works by randomly partitioning features.
    Anomalies are isolated in fewer splits → shorter path length → higher score.
    Score returned: decision_function output (negative = more anomalous).
    We flip sign so higher score = more anomalous.
    """

    def __init__(self, bearing_id: str, n_estimators: int = N_ESTIMATORS,
                 contamination: float = CONTAMINATION):
        self.bearing_id = bearing_id
        self.scaler = StandardScaler()
        self.model  = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.threshold_  = None
        self.feature_cols_ = None
        self.is_fitted_   = False

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only this bearing's feature columns."""
        cols = [c for c in df.columns if c.startswith(self.bearing_id)]
        self.feature_cols_ = cols
        return df[cols]

    def fit(self, df_normal: pd.DataFrame) -> "SingleBearingDetector":
        """
        Train on normal (healthy) snapshots only.

        Args:
            df_normal: feature matrix rows corresponding to healthy period
        """
        X = self._select_features(df_normal).values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        # Threshold = 99th percentile of training anomaly scores
        train_scores = -self.model.decision_function(X_scaled)
        self.threshold_ = float(np.percentile(train_scores, ANOMALY_PERCENTILE))
        self.is_fitted_ = True

        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Anomaly score for every snapshot in df.
        Higher = more anomalous. Normal range: near 0.
        """
        if not self.is_fitted_:
            raise RuntimeError(f"Detector for {self.bearing_id} not fitted yet.")
        X = df[self.feature_cols_].values
        X_scaled = self.scaler.transform(X)
        return -self.model.decision_function(X_scaled)

    def flag(self, scores: np.ndarray) -> np.ndarray:
        """Binary flag: 1 = anomaly (score > threshold), 0 = normal."""
        return (scores > self.threshold_).astype(int)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "SingleBearingDetector":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Multi-Bearing Detector ────────────────────────────────────────────────────

class BearingAnomalyDetector:
    """
    Manages one SingleBearingDetector per bearing.

    Trains on the first n_normal snapshots of each bearing independently,
    then produces anomaly score time-series for the full dataset.

    Args:
        n_normal     : number of healthy snapshots to train on
        n_estimators : Isolation Forest tree count
        contamination: expected fraction of anomalies in training window
    """

    def __init__(self, n_normal: int = N_NORMAL_DEFAULT,
                 n_estimators: int = N_ESTIMATORS,
                 contamination: float = CONTAMINATION):
        self.n_normal      = n_normal
        self.n_estimators  = n_estimators
        self.contamination = contamination
        self.detectors_: dict[str, SingleBearingDetector] = {}
        self.feature_matrix_: pd.DataFrame | None = None
        self.bearing_ids_: list[str] = []

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, feature_matrix_path: str) -> "BearingAnomalyDetector":
        """
        Load feature matrix CSV and fit one detector per bearing.

        Args:
            feature_matrix_path: path to CSV produced by features.py
        """
        print(f"Loading feature matrix: {feature_matrix_path}")
        df = pd.read_csv(feature_matrix_path, index_col=0)
        return self.fit_from_df(df)

    def fit_from_df(self, df: pd.DataFrame) -> "BearingAnomalyDetector":
        """Fit from an already-loaded feature matrix DataFrame."""
        self.feature_matrix_ = df

        # Infer bearing IDs from column prefixes (e.g. "b1_ch1_rms" → "b1_ch1")
        self.bearing_ids_ = sorted(set(
            "_".join(c.split("_")[:2]) for c in df.columns
        ))

        df_normal = df.iloc[:self.n_normal]
        print(f"Training on first {len(df_normal)} snapshots "
              f"(of {len(df)} total) per bearing")

        for bid in self.bearing_ids_:
            print(f"  Fitting detector: {bid} ...", end=" ")
            det = SingleBearingDetector(
                bearing_id=bid,
                n_estimators=self.n_estimators,
                contamination=self.contamination,
            )
            det.fit(df_normal)
            self.detectors_[bid] = det
            print(f"threshold={det.threshold_:.4f}")

        print(f"\n✓ All {len(self.bearing_ids_)} detectors fitted.")
        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_all(self) -> pd.DataFrame:
        """
        Score every snapshot for every bearing.

        Returns:
            DataFrame with columns: b1_ch1_score, b1_ch1_flag, b2_ch1_score ...
            Index = snapshot filename (same as feature_matrix_)
        """
        if not self.detectors_:
            raise RuntimeError("Call fit() before score_all().")

        results = {}
        for bid, det in self.detectors_.items():
            scores = det.score(self.feature_matrix_)
            flags  = det.flag(scores)
            results[f"{bid}_score"] = scores
            results[f"{bid}_flag"]  = flags

        df_out = pd.DataFrame(results, index=self.feature_matrix_.index)

        # Composite score: max across all bearings (system-level alert)
        score_cols = [c for c in df_out.columns if c.endswith("_score")]
        df_out["composite_score"] = df_out[score_cols].max(axis=1)
        df_out["composite_flag"]  = (df_out[score_cols] > 0).any(axis=1).astype(int)

        print(f"\nScoring complete.")
        for bid, det in self.detectors_.items():
            n_flagged = df_out[f"{bid}_flag"].sum()
            pct = 100 * n_flagged / len(df_out)
            print(f"  {bid}: {n_flagged} anomalies flagged ({pct:.1f}% of snapshots)")

        return df_out

    # ── Evaluation ────────────────────────────────────────────────────────────

    def find_first_alert(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        Find the first snapshot each bearing is flagged as anomalous.
        Useful for measuring early-detection lead time before failure.
        """
        alerts = {}
        flag_cols = [c for c in scores_df.columns if c.endswith("_flag")]
        for col in flag_cols:
            bid = col.replace("_flag", "")
            flagged = scores_df[scores_df[col] == 1]
            if not flagged.empty:
                first_idx = scores_df.index.get_loc(flagged.index[0])
                alerts[bid] = {
                    "first_alert_snapshot": flagged.index[0],
                    "snapshot_number":      first_idx,
                    "snapshots_before_end": len(scores_df) - first_idx,
                }
            else:
                alerts[bid] = {
                    "first_alert_snapshot": None,
                    "snapshot_number":      None,
                    "snapshots_before_end": None,
                }
        return pd.DataFrame(alerts).T

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot(self, scores_df: pd.DataFrame, output_path: str = None,
             show: bool = False) -> None:
        """
        Plot anomaly scores over time for each bearing.
        Red shading marks flagged anomaly regions.
        """
        n_bearings = len(self.bearing_ids_)
        fig, axes = plt.subplots(n_bearings, 1, figsize=(14, 3 * n_bearings),
                                 sharex=True)
        if n_bearings == 1:
            axes = [axes]

        x = np.arange(len(scores_df))

        for ax, bid in zip(axes, self.bearing_ids_):
            score_col = f"{bid}_score"
            flag_col  = f"{bid}_flag"
            threshold = self.detectors_[bid].threshold_

            scores = scores_df[score_col].values
            flags  = scores_df[flag_col].values

            ax.plot(x, scores, color="steelblue", linewidth=0.8, alpha=0.9,
                    label="Anomaly score")
            ax.axhline(threshold, color="orange", linewidth=1.2,
                       linestyle="--", label=f"Threshold ({threshold:.3f})")

            # Shade anomaly regions
            in_anomaly = False
            start = 0
            for i, flag in enumerate(flags):
                if flag == 1 and not in_anomaly:
                    in_anomaly = True
                    start = i
                elif flag == 0 and in_anomaly:
                    ax.axvspan(start, i, color="red", alpha=0.15)
                    in_anomaly = False
            if in_anomaly:
                ax.axvspan(start, len(flags), color="red", alpha=0.15)

            ax.set_ylabel("Anomaly score", fontsize=9)
            ax.set_title(f"{bid.upper()} — Isolation Forest", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Snapshot index (time →)", fontsize=9)
        fig.suptitle("Isolation Forest Anomaly Detection — NASA Bearing Dataset",
                     fontsize=12, y=1.01)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved → {output_path}")
        if show:
            plt.show()
        plt.close()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_models(self, directory: str) -> None:
        """Save all per-bearing detectors to disk."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        for bid, det in self.detectors_.items():
            path = os.path.join(directory, f"detector_{bid}.pkl")
            det.save(path)
        print(f"✓ Models saved → {directory}")

    def load_models(self, directory: str) -> None:
        """Load previously saved per-bearing detectors."""
        for path in Path(directory).glob("detector_*.pkl"):
            det = SingleBearingDetector.load(str(path))
            self.detectors_[det.bearing_id] = det
        print(f"✓ Loaded {len(self.detectors_)} models from {directory}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys 
    from isolation_forest import BearingAnomalyDetector, SingleBearingDetector
    from rul_lstm import BearingRULPredictor, SingleBearingRUL, LSTMRULModel

    if len(sys.argv) < 3:
        print("Usage: python isolation_forest.py <feature_matrix.csv> "
              "<output_dir> [n_normal]")
        print("  feature_matrix.csv : output from features.py")
        print("  output_dir         : where to save scores, flags, plots, models")
        print("  n_normal           : healthy training snapshots (default 500)")
        sys.exit(1)

    feature_csv = sys.argv[1]
    output_dir  = sys.argv[2]
    n_normal    = int(sys.argv[3]) if len(sys.argv) > 3 else N_NORMAL_DEFAULT

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    detector = BearingAnomalyDetector(n_normal=n_normal)
    detector.fit(feature_csv)

    scores_df = detector.score_all()
    scores_df.to_csv(os.path.join(output_dir, "anomaly_scores.csv"))
    print(f"Scores saved → {output_dir}/anomaly_scores.csv")

    alerts = detector.find_first_alert(scores_df)
    alerts.to_csv(os.path.join(output_dir, "first_alerts.csv"))
    print(f"First alerts saved → {output_dir}/first_alerts.csv")
    print("\nFirst alert summary:")
    print(alerts.to_string())

    detector.plot(scores_df,
                  output_path=os.path.join(output_dir, "anomaly_plot.png"))

    detector.save_models(os.path.join(output_dir, "models"))
