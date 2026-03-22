"""
signal_to_image.py — Vibration Signal to Image Converter
Industrial AI Predictive Maintenance | BearingMind

Converts raw bearing vibration signals into 64×64×3 images for the
CNN Autoencoder (cv_anomaly_detector.py) to train on.

Three conversion methods (stacked as RGB channels):
    Channel 0 — STFT Spectrogram  : frequency content over time
    Channel 1 — Mel Spectrogram   : frequency scaled to perceptual scale
    Channel 2 — Gramian Angular Field (GAF) : time-series correlations

Why convert to images?
    Raw vibration signals are 1D time-series of 20,480 samples.
    CNNs are designed for 2D spatial data.
    Spectrograms transform the 1D signal into a 2D (time × frequency)
    representation — the CNN can then learn spatial patterns in frequency
    space that correspond to bearing fault signatures (e.g. energy at BPFO).

Pipeline position:
    raw NASA IMS snapshots → signal_to_image.py
                           → images/normal/   (first N_NORMAL snapshots)
                           → images/all/      (all snapshots)
                           → cv_anomaly_detector.py

Output filenames:
    {snapshot_name}_{bearing_id}.npy
    e.g. 2004.02.12.10.32.39_b1_ch1.npy

Usage:
    from src.signal_to_image import SignalImageConverter

    converter = SignalImageConverter(method="stft", image_size=64)
    converter.process_dataset(
        data_path="path/to/2nd_test",
        output_dir="data/images",
        n_normal=500,
        n_channels=4
    )
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

# ── Optional imports (graceful fallback) ──────────────────────────────────────
try:
    from scipy import signal as scipy_signal
    from scipy.fft import fft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not installed. Install with: pip install scipy")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    # Mel spectrogram will fall back to STFT if librosa not available


# ── Constants ──────────────────────────────────────────────────────────────────

SAMPLING_RATE_HZ = 20_480       # NASA IMS test rig
IMAGE_SIZE       = 64           # output image: 64×64×3
N_NORMAL_DEFAULT = 500          # first N snapshots = healthy (for normal/ dir)
COLUMNS_4CH      = ["b1_ch1", "b2_ch1", "b3_ch1", "b4_ch1"]
COLUMNS_8CH      = ["b1_ch1", "b1_ch2", "b2_ch1", "b2_ch2",
                    "b3_ch1", "b3_ch2", "b4_ch1", "b4_ch2"]


# ── Individual Conversion Methods ─────────────────────────────────────────────

def signal_to_stft(signal: np.ndarray, fs: int = SAMPLING_RATE_HZ,
                   image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Short-Time Fourier Transform spectrogram.

    Slides a window across the signal and computes FFT at each position.
    Result: 2D array (frequency bins × time frames) showing how frequency
    content changes over time.

    Why STFT for bearings?
        Bearing faults create periodic impulses at characteristic frequencies
        (BPFO, BPFI, BSF). STFT shows these as horizontal bands of energy
        at specific frequency rows — the CNN learns to recognize these patterns.

    Args:
        signal     : raw vibration array (20480 samples)
        fs         : sampling rate in Hz
        image_size : output image dimensions (image_size × image_size)

    Returns:
        np.ndarray shape (image_size, image_size) — normalized to [0, 1]
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required. pip install scipy")

    nperseg = min(256, len(signal) // 8)
    noverlap = nperseg // 2

    _, _, Zxx = scipy_signal.stft(
        signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    magnitude = np.abs(Zxx)
    magnitude = np.log1p(magnitude)   # log scale — compresses dynamic range

    # Resize to image_size × image_size using simple interpolation
    img = _resize_2d(magnitude, image_size, image_size)
    img = _normalize(img)
    return img.astype(np.float32)


def signal_to_mel(signal: np.ndarray, fs: int = SAMPLING_RATE_HZ,
                  image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Mel-scale spectrogram.

    Like STFT but frequency axis is warped to the Mel scale — more bins
    at low frequencies where bearing fault energy concentrates, fewer at
    high frequencies. Better pattern separation for fault classification.

    Falls back to STFT if librosa is not installed.

    Args:
        signal     : raw vibration array
        fs         : sampling rate in Hz
        image_size : output image dimensions

    Returns:
        np.ndarray shape (image_size, image_size) — normalized to [0, 1]
    """
    if LIBROSA_AVAILABLE:
        n_fft   = min(512, len(signal) // 4)
        hop_len = n_fft // 4
        n_mels  = image_size

        mel = librosa.feature.melspectrogram(
            y=signal.astype(np.float32), sr=fs,
            n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        img = _resize_2d(mel_db, image_size, image_size)
        img = _normalize(img)
        return img.astype(np.float32)
    else:
        # Graceful fallback — use STFT if librosa not available
        return signal_to_stft(signal, fs, image_size)


def signal_to_gaf(signal: np.ndarray,
                  image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Gramian Angular Field (GAF).

    Encodes a time-series as a matrix of pairwise angular cosine similarities
    between time points. Preserves temporal correlations as spatial structure.

    Why GAF for bearings?
        Captures long-range temporal dependencies — e.g. the periodicity of
        fault impulses across the full signal window. STFT captures local
        frequency content; GAF captures global temporal patterns.
        Together they give the CNN complementary information.

    Method:
        1. Downsample signal to image_size points
        2. Normalise to [-1, 1]
        3. Convert to angular representation: φ = arccos(x)
        4. GAF[i,j] = cos(φ_i + φ_j)

    Args:
        signal     : raw vibration array
        image_size : output image dimensions (image_size × image_size)

    Returns:
        np.ndarray shape (image_size, image_size) — normalized to [0, 1]
    """
    # Downsample to image_size points
    indices = np.linspace(0, len(signal) - 1, image_size).astype(int)
    s = signal[indices].astype(np.float64)

    # Normalise to [-1, 1] — required for arccos
    s_min, s_max = s.min(), s.max()
    if s_max - s_min > 1e-10:
        s = 2.0 * (s - s_min) / (s_max - s_min) - 1.0
    else:
        s = np.zeros_like(s)

    # Clip for numerical safety before arccos
    s = np.clip(s, -1.0, 1.0)
    phi = np.arccos(s)   # angular representation

    # GAF: cos(φ_i + φ_j) for all pairs (i, j)
    phi_i = phi[:, np.newaxis]   # (N, 1)
    phi_j = phi[np.newaxis, :]   # (1, N)
    gaf = np.cos(phi_i + phi_j)  # (N, N)

    img = _normalize(gaf)
    return img.astype(np.float32)


def signal_to_rgb_image(signal: np.ndarray, fs: int = SAMPLING_RATE_HZ,
                         image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Combine STFT + Mel + GAF into a single 3-channel (RGB) image.

    Channel 0 (R) = STFT spectrogram
    Channel 1 (G) = Mel spectrogram
    Channel 2 (B) = Gramian Angular Field

    Why 3 channels?
        The CNN Autoencoder expects 64×64×3 input.
        Each channel carries different information about the same signal:
        - STFT: local frequency content over time
        - Mel: perceptually-scaled frequency content
        - GAF: global temporal correlations
        Together they give the CNN richer information to learn from.

    Args:
        signal     : raw vibration array from one bearing snapshot
        fs         : sampling rate in Hz
        image_size : output H and W dimensions

    Returns:
        np.ndarray shape (image_size, image_size, 3) — float32 in [0, 1]
    """
    ch0 = signal_to_stft(signal, fs, image_size)
    ch1 = signal_to_mel(signal,  fs, image_size)
    ch2 = signal_to_gaf(signal,      image_size)

    return np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resize_2d(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize a 2D array to (target_h, target_w) using nearest-neighbour
    index mapping. Avoids PIL/cv2 dependency — only numpy needed.
    """
    h, w = arr.shape
    row_idx = np.linspace(0, h - 1, target_h).astype(int)
    col_idx = np.linspace(0, w - 1, target_w).astype(int)
    return arr[np.ix_(row_idx, col_idx)]


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0, 1]."""
    a_min, a_max = arr.min(), arr.max()
    if a_max - a_min > 1e-10:
        return (arr - a_min) / (a_max - a_min)
    return np.zeros_like(arr, dtype=np.float32)


# ── Main Converter Class ───────────────────────────────────────────────────────

class SignalImageConverter:
    """
    Converts NASA IMS bearing snapshots to 64×64×3 spectrogram images.

    Reads raw snapshot files → converts each bearing's signal →
    saves as .npy arrays in organised output directories.

    Output structure:
        output_dir/
            normal/    ← first n_normal snapshots (healthy bearings)
            all/       ← every snapshot (for scoring the full run)

    Args:
        image_size : output image H and W (default 64)
        fs         : sampling rate in Hz (default 20480)
    """

    def __init__(self, image_size: int = IMAGE_SIZE,
                 fs: int = SAMPLING_RATE_HZ):
        self.image_size = image_size
        self.fs         = fs

    def convert_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Convert one bearing's raw signal to a 64×64×3 image.

        Args:
            signal : 1D vibration array from one snapshot/bearing

        Returns:
            np.ndarray shape (image_size, image_size, 3) float32 in [0,1]
        """
        return signal_to_rgb_image(signal, self.fs, self.image_size)

    def _load_snapshot(self, filepath: str,
                       columns: list) -> pd.DataFrame:
        """Load one raw NASA IMS snapshot file."""
        return pd.read_csv(
            filepath, sep="\t", header=None, names=columns)

    def process_dataset(self, data_path: str, output_dir: str,
                        n_normal: int = N_NORMAL_DEFAULT,
                        n_channels: int = 4,
                        verbose: bool = True) -> dict:
        """
        Process every snapshot file in data_path and save images.

        Creates two directories:
            output_dir/normal/ — first n_normal snapshots (healthy)
            output_dir/all/    — all snapshots

        Naming convention:
            {snapshot_filename}_{bearing_id}.npy

        Args:
            data_path  : directory containing raw NASA IMS snapshot files
            output_dir : root directory to save images
            n_normal   : number of healthy snapshots for normal/ directory
            n_channels : 4 for 2nd/3rd test, 8 for 1st test
            verbose    : print progress

        Returns:
            dict with keys 'normal_count', 'all_count', 'bearing_ids'
        """
        columns     = COLUMNS_4CH if n_channels == 4 else COLUMNS_8CH
        bearing_ids = [c for c in columns if c.endswith("ch1")]

        normal_dir = Path(output_dir) / "normal"
        all_dir    = Path(output_dir) / "all"
        normal_dir.mkdir(parents=True, exist_ok=True)
        all_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            f for f in os.listdir(data_path) if not f.startswith(".")
        )
        if not files:
            raise FileNotFoundError(f"No files found in: {data_path}")

        if verbose:
            print(f"Found {len(files)} snapshot files")
            print(f"Bearings: {bearing_ids}")
            print(f"Normal threshold: first {n_normal} snapshots → normal/")
            print(f"Output: {output_dir}\n")

        normal_count = 0
        all_count    = 0

        for i, fname in enumerate(files):
            if verbose and i % 100 == 0:
                print(f"  [{i+1:4d}/{len(files)}] {fname}")

            try:
                filepath = os.path.join(data_path, fname)
                df       = self._load_snapshot(filepath, columns)
                is_normal = (i < n_normal)

                for bid in bearing_ids:
                    signal   = df[bid].values.astype(np.float32)
                    img      = self.convert_signal(signal)
                    img_name = f"{fname}_{bid}.npy"

                    # Always save to all/
                    np.save(str(all_dir / img_name), img)
                    all_count += 1

                    # Save to normal/ only for healthy snapshots
                    if is_normal:
                        np.save(str(normal_dir / img_name), img)
                        normal_count += 1

            except Exception as e:
                if verbose:
                    print(f"  Warning: skipped {fname} — {e}")

        if verbose:
            print(f"\n✓ Done.")
            print(f"  normal/ : {normal_count} images "
                  f"({n_normal} snapshots × {len(bearing_ids)} bearings)")
            print(f"  all/    : {all_count} images "
                  f"({len(files)} snapshots × {len(bearing_ids)} bearings)")

        return {
            "normal_count": normal_count,
            "all_count":    all_count,
            "bearing_ids":  bearing_ids,
            "normal_dir":   str(normal_dir),
            "all_dir":      str(all_dir),
        }

    def process_single_snapshot(self, filepath: str,
                                 n_channels: int = 4) -> dict:
        """
        Convert one snapshot file to images for all bearings.
        Used by the orchestrator agent for real-time single-snapshot scoring.

        Returns:
            dict mapping bearing_id → np.ndarray (image_size, image_size, 3)
        """
        columns     = COLUMNS_4CH if n_channels == 4 else COLUMNS_8CH
        bearing_ids = [c for c in columns if c.endswith("ch1")]
        df          = self._load_snapshot(filepath, columns)

        return {
            bid: self.convert_signal(df[bid].values.astype(np.float32))
            for bid in bearing_ids
        }

    # ── Visualisation ─────────────────────────────────────────────────────────

    def visualize_channels(self, signal: np.ndarray,
                           bearing_id: str = "b1_ch1",
                           output_path: str = None) -> None:
        """
        Plot all three image channels side by side for one signal.
        Useful for verifying conversion is working correctly.

        Args:
            signal      : raw vibration array
            bearing_id  : label for plot title
            output_path : save path (None = show only)
        """
        ch0 = signal_to_stft(signal, self.fs, self.image_size)
        ch1 = signal_to_mel(signal,  self.fs, self.image_size)
        ch2 = signal_to_gaf(signal,           self.image_size)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(ch0, cmap="viridis", aspect="auto")
        axes[0].set_title("Ch0: STFT Spectrogram", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(ch1, cmap="viridis", aspect="auto")
        axes[1].set_title("Ch1: Mel Spectrogram", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(ch2, cmap="viridis", aspect="auto")
        axes[2].set_title("Ch2: Gramian Angular Field", fontsize=10)
        axes[2].axis("off")

        # Combined RGB
        rgb = np.stack([ch0, ch1, ch2], axis=-1)
        axes[3].imshow(rgb, aspect="auto")
        axes[3].set_title("Combined RGB (CNN input)", fontsize=10)
        axes[3].axis("off")

        fig.suptitle(
            f"Signal-to-Image Conversion — {bearing_id}", fontsize=12)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved → {output_path}")
        else:
            plt.show()
        plt.close()

    def visualize_sample_grid(self, image_dir: str, n_samples: int = 8,
                               output_path: str = None) -> None:
        """
        Show a grid of sample images from a directory.
        Useful for confirming images look sensible before training.
        """
        paths = sorted(Path(image_dir).glob("*.npy"))[:n_samples]
        if not paths:
            print(f"No .npy files found in {image_dir}")
            return

        fig, axes = plt.subplots(1, len(paths),
                                  figsize=(3 * len(paths), 3))
        if len(paths) == 1:
            axes = [axes]

        for ax, p in zip(axes, paths):
            img = np.load(str(p))
            ax.imshow(img, aspect="auto")
            ax.set_title(p.stem[-20:], fontsize=6)
            ax.axis("off")

        fig.suptitle(f"Sample images from {Path(image_dir).name}/",
                     fontsize=10)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Grid saved → {output_path}")
        else:
            plt.show()
        plt.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python signal_to_image.py <data_path> <output_dir> "
              "[n_normal] [n_channels]")
        print("  data_path  : directory with NASA IMS snapshot files")
        print("  output_dir : root output directory (normal/ and all/ created here)")
        print("  n_normal   : healthy snapshot count for normal/ dir (default 500)")
        print("  n_channels : 4 for 2nd/3rd test, 8 for 1st test (default 4)")
        sys.exit(1)

    data_path  = sys.argv[1]
    output_dir = sys.argv[2]
    n_normal   = int(sys.argv[3]) if len(sys.argv) > 3 else N_NORMAL_DEFAULT
    n_channels = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    data_path="E:\\Alok\\Job_Prep\\Projects_and_Codes\\nasa_archive\\2nd_test\\2nd_test"

    converter = SignalImageConverter()
    stats = converter.process_dataset(
        data_path=data_path,
        output_dir=output_dir,
        n_normal=n_normal,
        n_channels=n_channels,
    )

    print(f"\nRun cv_anomaly_detector.py next:")
    print(f"  python src/cv_anomaly_detector.py "
          f"{stats['normal_dir']} {stats['all_dir']} results/cv/ 50")