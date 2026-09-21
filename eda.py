"""
wesad_eda.py — Exploratory Data Analysis for the WESAD dataset (chest signals)

Loads each subject's raw WESAD .pkl file, segments the chest-worn RespiBAN
signals (ECG, EDA, Resp, Temp) into 60-second windows, extracts the 10
clinically-motivated features, and reproduces 4 EDA charts:

    1. Grouped bar chart  — windows per subject, Baseline vs Distress
    2. Ridgeline plot     — HR/SDNN/RMSSD/EDA_Mean/EDA_Std/EDA_Peaks by
                             condition, with Mann-Whitney U significance stars
    3. Correlation heatmap of all extracted features
    4. Pie chart of class proportion / imbalance ratio

IMPORTANT — sampling rate correction
-------------------------------------
The WESAD chest device (RespiBAN) samples ECG, EDA, EMG, Resp, Temp and ACC
ALL at 700 Hz. The commonly-cited "EDA @ 4Hz" figure applies only to the
WRIST-worn Empatica E4 device, not the chest device used here. This script
uses FS = 700 for every chest channel including EDA. If other parts of your
project (app.py, your paper's Algorithm 1) assume 4Hz for chest EDA, that
assumption is inconsistent with the actual WESAD chest data and is worth
revisiting — it changes the scale of EDA_Peaks/arousal_index by ~175x
(700/4) since the window contains far more raw samples than assumed.

Install dependencies:
    pip install numpy pandas scipy matplotlib seaborn neurokit2
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks

try:
    import neurokit2 as nk
    HAVE_NK = True
except ImportError:
    HAVE_NK = False
    print("neurokit2 not found -- falling back to scipy-based peak detection.")
    print("For best results: pip install neurokit2")


# ============================================================
# DASHBOARD DARK THEME (matches the project's existing chart style)
# ============================================================
BG = "#0B0F1A"          # outer figure background
PANEL_BG = "#111827"    # inner plot-panel background
GRID = "#1E293B"        # faint gridlines
TEXT = "#E5E7EB"        # titles
TEXT_MUTED = "#94A3B8"  # axis ticks / muted labels

BLUE = "#38BDF8"        # accent blue -- Emotional Baseline
RED = "#F87171"         # accent red  -- Emotional Distress
BLUE_BAR = "#2D90BF"    # bar-chart blue (slightly deeper, matches reference)
RED_BAR = "#B9595D"     # bar-chart red

COND_COLORS = {"Emotional Baseline": BLUE, "Emotional Distress": RED}
COND_BAR_COLORS = {"Emotional Baseline": BLUE_BAR, "Emotional Distress": RED_BAR}


def _style_dark_axes(ax, panel=True):
    """Apply the dashboard dark theme to one Axes."""
    ax.set_facecolor(PANEL_BG if panel else BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)
    ax.title.set_color(TEXT)


# ============================================================
# CONFIGURATION
# ============================================================

# Fixed: raw string (r"...") avoids the \N unicode-escape SyntaxError caused
# by a plain string interpreting "\N" as an escape sequence on Windows paths.
WESAD_DIR = Path(r"C:\NeuroWatch\Data\WESAD")
OUTPUT_DIR = Path(r"C:\NeuroWatch\Data\WESAD_EDA_Output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 700  # Hz -- shared by ALL chest signals (ECG, EDA, Resp, Temp). See note above.
WINDOW_SEC = 60
WINDOW_SAMPLES = FS * WINDOW_SEC

LABEL_BASELINE = 1
LABEL_STRESS = 2
VALID_LABELS = {LABEL_BASELINE, LABEL_STRESS}
MIN_LABEL_PURITY = 0.90  # window kept only if >=90% of samples share one label

# Standard WESAD convention: S1 (pilot) and S12 (protocol deviation) excluded.
# Flag / edit this if it doesn't match your copy of the dataset.
EXCLUDED_SUBJECTS = {"S1", "S12"}
SUBJECTS = [f"S{i}" for i in range(2, 18) if f"S{i}" not in EXCLUDED_SUBJECTS]


# ============================================================
# LOADING
# ============================================================

def load_subject(subject_id):
    """Load one WESAD subject's chest signals + labels from its .pkl file."""
    pkl_path = WESAD_DIR / subject_id / f"{subject_id}.pkl"
    if not pkl_path.exists():
        print(f"  [skip] {pkl_path} not found")
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    chest = data["signal"]["chest"]
    return {
        "ecg": np.asarray(chest["ECG"]).flatten(),
        "eda": np.asarray(chest["EDA"]).flatten(),
        "resp": np.asarray(chest["Resp"]).flatten(),
        "temp": np.asarray(chest["Temp"]).flatten(),
        "label": np.asarray(data["label"]).flatten(),
    }


# ============================================================
# WINDOWING
# ============================================================

def make_windows(n_samples, window_samples):
    """Non-overlapping (start, end) index pairs."""
    n_windows = n_samples // window_samples
    return [(i * window_samples, (i + 1) * window_samples) for i in range(n_windows)]


def window_label(label_slice):
    """Majority label if it covers >= MIN_LABEL_PURITY of the window, else None."""
    vals, counts = np.unique(label_slice, return_counts=True)
    majority_idx = np.argmax(counts)
    majority_label = vals[majority_idx]
    purity = counts[majority_idx] / len(label_slice)
    if majority_label in VALID_LABELS and purity >= MIN_LABEL_PURITY:
        return int(majority_label)
    return None


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_ecg_features(ecg_window, fs=FS):
    rpeaks = np.array([])
    if HAVE_NK:
        try:
            _, info = nk.ecg_process(ecg_window, sampling_rate=fs)
            rpeaks = np.asarray(info["ECG_R_Peaks"])
        except Exception:
            rpeaks = np.array([])

    if len(rpeaks) < 2:
        rpeaks, _ = find_peaks(ecg_window, distance=fs * 0.4)

    if len(rpeaks) < 2:
        return {"HR": 70.0, "SDNN": 0.05, "RMSSD": 0.04}  # fallback defaults

    rr = np.diff(rpeaks) / fs  # seconds
    hr = 60.0 / np.mean(rr)
    sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 2 else 0.0
    return {"HR": hr, "SDNN": sdnn, "RMSSD": rmssd}


def extract_eda_features(eda_window, fs=FS):
    eda_mean = float(np.mean(eda_window))
    eda_std = float(np.std(eda_window, ddof=1))

    peaks = None
    if HAVE_NK:
        try:
            signals, _ = nk.eda_process(eda_window, sampling_rate=fs)
            peaks = np.where(signals["SCR_Peaks"] == 1)[0]
        except Exception:
            peaks = None

    if peaks is None:
        threshold = eda_mean + 0.1 * eda_std
        peaks, _ = find_peaks(eda_window, height=threshold, distance=fs)  # >=1s apart

    n_peaks = min(len(peaks), 20)  # clipped
    arousal_index = n_peaks / len(eda_window)
    return {"EDA_Mean": eda_mean, "EDA_Std": eda_std, "EDA_Peaks": n_peaks, "AI": arousal_index}


def extract_resp_temp_features(resp_window, temp_window):
    return {
        "Resp_Mean": float(np.mean(resp_window)),
        "Resp_Std": float(np.std(resp_window, ddof=1)),
        "Temp_Mean": float(np.mean(temp_window)),
    }


# ============================================================
# MAIN EXTRACTION PIPELINE
# ============================================================

def process_subject(subject_id):
    raw = load_subject(subject_id)
    if raw is None:
        return []

    n_samples = min(len(raw["ecg"]), len(raw["eda"]), len(raw["resp"]),
                     len(raw["temp"]), len(raw["label"]))
    windows = make_windows(n_samples, WINDOW_SAMPLES)

    rows = []
    for start, end in windows:
        label = window_label(raw["label"][start:end])
        if label is None:
            continue  # transition / non baseline-or-stress window, dropped

        row = {
            "Subject": subject_id,
            "Condition": "Emotional Distress" if label == LABEL_STRESS else "Emotional Baseline",
        }
        row.update(extract_ecg_features(raw["ecg"][start:end]))
        row.update(extract_eda_features(raw["eda"][start:end]))
        row.update(extract_resp_temp_features(raw["resp"][start:end], raw["temp"][start:end]))
        rows.append(row)
    return rows


def build_features_dataframe():
    all_rows = []
    for subject_id in SUBJECTS:
        print(f"Processing {subject_id}...")
        all_rows.extend(process_subject(subject_id))

    df = pd.DataFrame(all_rows)
    if not df.empty:
        # Illustrative composite column for the heatmap ONLY -- not a real
        # WESAD measurement. Replace with a genuine arousal/self-report
        # column if you have one.
        df["Arousal"] = (stats.zscore(df["HR"]) + stats.zscore(df["EDA_Mean"])) / 2
    return df


# ============================================================
# PLOT 1 — grouped bar chart
# ============================================================

def plot_window_counts(df, out_path):
    counts = df.groupby(["Subject", "Condition"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["Emotional Baseline", "Emotional Distress"], fill_value=0)
    # keep WESAD's natural subject order (S2, S3, ... S17) rather than alphabetical
    counts = counts.reindex(index=[s for s in SUBJECTS if s in counts.index])

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    _style_dark_axes(ax, panel=True)

    x = np.arange(len(counts))
    width = 0.38
    bars_b = ax.bar(x - width / 2, counts["Emotional Baseline"], width,
                     color=BLUE_BAR, label="Emotional Baseline")
    bars_d = ax.bar(x + width / 2, counts["Emotional Distress"], width,
                     color=RED_BAR, label="Emotional Distress")

    for bars in (bars_b, bars_d):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.8, f"{int(h)}",
                        ha="center", va="bottom", fontsize=7, color=TEXT_MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, color=TEXT_MUTED, fontsize=9)
    ax.set_ylabel("Number of 60-second Windows", fontsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    legend = ax.legend(facecolor=PANEL_BG, edgecolor="none", labelcolor=TEXT_MUTED,
                        loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=BG)
    plt.close()


# ============================================================
# PLOT 2 — ridgeline plot with Mann-Whitney U significance stars
# ============================================================

RIDGELINE_LABELS = {
    "HR": "Heart Rate (bpm)", "SDNN": "SDNN (s)", "RMSSD": "RMSSD (s)",
    "EDA_Mean": "EDA Mean (\u00b5S)", "EDA_Std": "EDA Std (\u00b5S)", "EDA_Peaks": "EDA Peaks",
}


def plot_ridgeline(df, features, out_path):
    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(9, 0.95 * n), sharex=False)
    fig.patch.set_facecolor(BG)
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        _style_dark_axes(ax, panel=False)
        base_vals = df.loc[df.Condition == "Emotional Baseline", feat].dropna()
        stress_vals = df.loc[df.Condition == "Emotional Distress", feat].dropna()

        for vals, cond in [(base_vals, "Emotional Baseline"), (stress_vals, "Emotional Distress")]:
            color = COND_COLORS[cond]
            if len(vals) > 1 and vals.std() > 0:
                kde = stats.gaussian_kde(vals)
                pad = (vals.max() - vals.min()) * 0.25 + 1e-9
                x = np.linspace(vals.min() - pad, vals.max() + pad, 300)
                y = kde(x)
                ax.fill_between(x, y, color=color, alpha=0.32, zorder=2)
                ax.plot(x, y, color=color, linewidth=1.6, zorder=3)
                peak_x = x[np.argmax(y)]
                ax.axvline(peak_x, color=color, linestyle=(0, (3, 3)),
                           linewidth=1, alpha=0.85, zorder=4)

        stars = "ns"
        if len(base_vals) > 1 and len(stress_vals) > 1:
            _, p = stats.mannwhitneyu(base_vals, stress_vals, alternative="two-sided")
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        ax.set_ylabel(RIDGELINE_LABELS.get(feat, feat), rotation=0, ha="right", va="center",
                       fontsize=9.5, color=TEXT_MUTED, labelpad=12)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(1.01, 0.5, stars, transform=ax.transAxes, ha="left", va="center",
                 fontsize=10, fontweight="bold", color=RED)

    axes[0].plot([], [], color=BLUE, linewidth=2, label="Emotional Baseline")
    axes[0].plot([], [], color=RED, linewidth=2, label="Emotional Distress")
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 1.6), fontsize=8,
                    frameon=False, labelcolor=TEXT_MUTED, ncol=2)

    fig.subplots_adjust(left=0.16, right=0.94, top=0.90, bottom=0.03, hspace=0.15)
    plt.savefig(out_path, dpi=150, facecolor=BG)
    plt.close()


# ============================================================
# PLOT 3 — correlation heatmap
# ============================================================

def plot_correlation_heatmap(df, out_path):
    cols = ["HR", "SDNN", "RMSSD", "EDA_Mean", "EDA_Std", "EDA_Peaks",
            "Arousal", "Resp_Mean", "Resp_Std", "Temp_Mean"]
    labels = ["HR", "SDNN", "RMSSD", "EDA\nMean", "EDA\nStd", "EDA\nPeaks",
              "Arousal", "Resp\nMean", "Resp\nStd", "Temp"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8.5, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    hm = sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                      square=True, linewidths=1, linecolor=BG,
                      annot_kws={"fontsize": 8, "color": "#0B0F1A"},
                      cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"}, ax=ax)

    ax.set_xticklabels(labels, color=TEXT_MUTED, fontsize=8.5, rotation=0)
    ax.set_yticklabels(labels, color=TEXT_MUTED, fontsize=8.5, rotation=0)
    ax.set_title("Correlation Matrix of Extracted Features", color=TEXT, fontsize=12, pad=14)

    cbar = hm.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TEXT_MUTED)
    cbar.ax.tick_params(colors=TEXT_MUTED)
    cbar.outline.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=BG)
    plt.close()


# ============================================================
# PLOT 4 — pie chart of class imbalance
# ============================================================

def plot_class_proportion(df, out_path):
    counts = df["Condition"].value_counts().reindex(
        ["Emotional Baseline", "Emotional Distress"]).fillna(0)
    ratio = counts.max() / counts.min()
    colors = [BLUE, RED]
    pie_labels = ["Emotional Baseline\n(Non-Stress)", "Emotional Distress\n(Stress)"]

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    wedges, _, autotexts = ax.pie(
        counts.values, colors=colors, startangle=90, counterclock=False,
        autopct="%1.1f%%", pctdistance=0.68,
        wedgeprops={"edgecolor": BG, "linewidth": 2},
        textprops={"color": "#0B1220", "fontsize": 13, "fontweight": "bold"})

    for i, w in enumerate(wedges):
        ang = (w.theta1 + w.theta2) / 2
        x, y = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        ax.annotate(pie_labels[i], xy=(x * 0.98, y * 0.98), xytext=(x * 1.28, y * 1.18),
                    ha="center", va="center", fontsize=9.5, color=TEXT,
                    arrowprops=None)

    ax.set_title(f"Class Proportion\n({ratio:.1f}:1 Imbalance Ratio)",
                 color=TEXT, fontsize=13, pad=18)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=BG)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Reading WESAD from: {WESAD_DIR}")
    df = build_features_dataframe()

    if df.empty:
        print("No windows extracted -- check WESAD_DIR and subject folder names.")
        return

    csv_path = OUTPUT_DIR / "wesad_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved features: {csv_path}  ({len(df)} windows, {df['Subject'].nunique()} subjects)")

    plot_window_counts(df, OUTPUT_DIR / "1_window_counts_per_subject.png")
    plot_ridgeline(df, ["HR", "SDNN", "RMSSD", "EDA_Mean", "EDA_Std", "EDA_Peaks"],
                   OUTPUT_DIR / "2_ridgeline_features_by_condition.png")
    plot_correlation_heatmap(df, OUTPUT_DIR / "3_correlation_heatmap.png")
    plot_class_proportion(df, OUTPUT_DIR / "4_class_proportion_pie.png")
    print(f"All 4 plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()