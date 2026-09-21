"""
NeuroWatch — Phase 2 Watch Simulator
──────────────────────────────────────────────────────────────────────────
Stands in for a real wearable device. Generates a synthetic physiological
feature window (matching the exact 10-feature schema app.py trains on)
roughly once per interval and POSTs it to api.py's /ingest endpoint.

Run (after api.py is already running):
    python watch_simulation.py
    python watch_simulation.py --interval 2 --minutes 30 --patient-id S2
    python watch_simulation.py --stress-every 8

Calibration:
    If app.py has pushed calibration stats to the API (sidebar button
    "📡 Push Calibration to Live API" after training), this simulator pulls
    them on startup — and re-checks periodically (see --recalibrate-every)
    so pushing calibration mid-run takes effect without restarting the
    simulator. It then drifts around the *actual* per-feature WESAD
    mean/std, so live predictions line up with what the model saw in
    training. Otherwise it falls back to generic resting-patient defaults
    below — good enough for a demo, but not calibrated to any real subject.

Feel of a real watch:
    A real wearable doesn't hold perfectly still between readings — HR
    ticks up and down breath to breath, EDA has a faint continuous ripple.
    generate_window() layers a fast small-amplitude wobble (mostly on
    heart_rate) on top of the slower drift so consecutive pushes visibly
    move instead of looking frozen, while --interval controls how often a
    reading actually reaches the dashboard.
"""

import argparse
import math
import random
import time
from datetime import datetime

import requests

FEATURES = [
    "heart_rate", "sdnn", "rmssd",
    "eda_mean", "eda_std", "eda_peaks", "arousal_index",
    "resp_mean", "resp_std", "temp_mean",
]

# Generic resting-patient fallback, used only when no calibration is
# available from the API. Real WESAD per-subject numbers will differ.
DEFAULT_BASELINE = {
    "heart_rate":    {"mean": 72.0, "std": 6.0},
    "sdnn":          {"mean": 0.07, "std": 0.02},
    "rmssd":         {"mean": 0.05, "std": 0.015},
    "eda_mean":      {"mean": 2.5,  "std": 1.2},
    "eda_std":       {"mean": 0.3,  "std": 0.15},
    "eda_peaks":     {"mean": 2.0,  "std": 1.5},
    "arousal_index": {"mean": 0.01, "std": 0.006},
    "resp_mean":     {"mean": 0.0,  "std": 0.2},
    "resp_std":      {"mean": 0.15, "std": 0.05},
    "temp_mean":     {"mean": 33.5, "std": 1.0},
}

# Applied on top of baseline during a simulated stress episode. Only
# features that plausibly move under acute stress get a multiplier.
STRESS_MULTIPLIER = {
    "heart_rate":    1.22,
    "sdnn":          0.55,
    "rmssd":         0.5,
    "eda_mean":      1.9,
    "eda_std":       1.8,
    "eda_peaks":     2.6,
    "arousal_index": 2.6,
    "resp_std":      1.6,
}


def fetch_calibration(api_url: str) -> dict:
    try:
        r = requests.get(f"{api_url}/calibration", timeout=3)
        if r.status_code == 200:
            print("✅ Loaded calibration stats from live API.")
            return r.json()
    except requests.exceptions.RequestException:
        pass
    print("⚠️  No calibration available — using generic resting-patient defaults. "
          "(Train the model in app.py, then click 'Push Calibration to Live API'.)")
    return DEFAULT_BASELINE


# Features that get a fast, small-amplitude wobble layered on top of the
# slower drift, so back-to-back readings visibly move like a live sensor
# instead of a flat average. Kept out of eda_peaks/arousal_index since
# those are integer-ish/derived and look wrong with sub-unit jitter.
WOBBLE_FEATURES = {"heart_rate", "eda_mean", "resp_mean", "resp_std", "sdnn", "rmssd"}


def generate_window(baseline: dict, drift: dict, stressed: bool, tick: int) -> dict:
    """One synthetic feature window: a random walk around each feature's
    baseline mean (drift) plus a faster small wobble (breath/beat-to-beat
    style variation), with an optional stress multiplier."""
    reading = {}
    for feat in FEATURES:
        stats = baseline.get(feat, DEFAULT_BASELINE[feat])
        mean, std = float(stats["mean"]), float(stats["std"])

        # Correlated drift: reacts faster than before (lower smoothing,
        # larger step) so the simulated patient's baseline visibly moves
        # over the course of a session rather than sitting still.
        drift[feat] = 0.6 * drift.get(feat, 0.0) + random.gauss(0, std * 0.7)
        value = mean + drift[feat]

        if feat in WOBBLE_FEATURES:
            # Fast per-reading wobble on top of the slower drift — sinusoid
            # with a bit of phase noise plus a small random jitter, so two
            # consecutive readings are never identical even mid-drift.
            wobble = std * 0.18 * math.sin(tick * 0.9 + hash(feat) % 10)
            value += wobble + random.gauss(0, std * 0.1)

        if stressed and feat in STRESS_MULTIPLIER:
            value *= STRESS_MULTIPLIER[feat]

        if feat == "eda_peaks":
            value = max(0, round(value))
        if feat in ("heart_rate", "eda_mean", "eda_std", "sdnn", "rmssd", "resp_std"):
            value = max(0.001, value)

        reading[feat] = round(float(value), 4)
    return reading


def main():
    ap = argparse.ArgumentParser(description="NeuroWatch wearable simulator")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--interval", type=float, default=2.0,
                     help="Seconds between simulated readings (default 2s so the "
                          "dashboard feels live; a real device would send one "
                          "every ~60s of real time).")
    ap.add_argument("--minutes", type=int, default=0,
                     help="Number of readings to send (0 = run forever).")
    ap.add_argument("--patient-id", default="S2")
    ap.add_argument("--stress-every", type=int, default=10,
                     help="Trigger a ~3-reading stress episode roughly every "
                          "N readings (0 = never simulate stress).")
    ap.add_argument("--recalibrate-every", type=int, default=15,
                     help="Re-check the API for updated calibration every N "
                          "readings, so pushing calibration mid-run takes "
                          "effect without restarting this script (0 = only "
                          "check once at startup).")
    args = ap.parse_args()

    baseline = fetch_calibration(args.api_url)
    drift = {}
    window_count = 0
    stress_remaining = 0

    print(f"📡 Streaming to {args.api_url}/ingest every {args.interval}s "
          f"(patient={args.patient_id}, stress_every={args.stress_every}, "
          f"recalibrate_every={args.recalibrate_every})")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            window_count += 1

            if (args.recalibrate_every and window_count > 1
                    and window_count % args.recalibrate_every == 0):
                baseline = fetch_calibration(args.api_url)

            if args.stress_every and window_count % args.stress_every == 0:
                stress_remaining = 3  # short simulated distress episode

            stressed = stress_remaining > 0
            if stressed:
                stress_remaining -= 1

            reading = generate_window(baseline, drift, stressed, window_count)
            reading["patient_id"] = args.patient_id
            reading["source"] = "watch_simulation"
            reading["timestamp"] = datetime.now().isoformat()

            try:
                r = requests.post(f"{args.api_url}/ingest", json=reading, timeout=3)
                r.raise_for_status()
                tag = "🔴 STRESS" if stressed else "🟢 stable "
                buffered = r.json().get("buffered_readings")
                print(f"[{window_count:04d}] {tag} · HR={reading['heart_rate']:.1f} "
                      f"EDA={reading['eda_mean']:.2f} → buffered={buffered}")
            except requests.exceptions.RequestException as e:
                print(f"[{window_count:04d}] ⚠️  Could not reach API: {e}")

            if args.minutes and window_count >= args.minutes:
                break
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")


if __name__ == "__main__":
    main()
