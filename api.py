"""
NeuroWatch — Phase 2 Live API
──────────────────────────────────────────────────────────────────────────
FastAPI backend that receives streamed physiological feature windows from
a wearable device (or the watch_simulation.py stand-in) and serves them to
the Streamlit dashboard's "Live Watch" tab.

Run (separate terminal, before or after app.py — order doesn't matter):
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Design note:
This service intentionally does NOT run the ML model. app.py already owns
the trained RandomForestClassifier + StandardScaler inside its Streamlit
session (loaded once via @st.cache_resource), so keeping prediction there
avoids re-loading/re-training a second copy of the model in this process.
This API is a thin, in-memory buffer/relay:

    watch_simulation.py  --POST /ingest-->  api.py (this file)
    app.py "Live Watch" tab  --GET /latest or /history-->  api.py
    app.py then calls its own predict_minute() on whatever it receives.

The in-memory buffer resets whenever this process restarts — that's fine
for a demo/dev setup. For a real deployment you'd swap the deque for
Redis/Postgres, but the endpoint contract below would stay the same.
"""

from collections import deque
from datetime import datetime
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Must match FEATURES in app.py exactly (same 10 names). Order doesn't
# matter here — app.py rebuilds a DataFrame keyed by name before scaling.
FEATURES = [
    "heart_rate", "sdnn", "rmssd",
    "eda_mean", "eda_std", "eda_peaks", "arousal_index",
    "resp_mean", "resp_std", "temp_mean",
]

MAX_BUFFER = 500  # plenty for a demo session; oldest readings drop off

app = FastAPI(title="NeuroWatch Live API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────
class WatchReading(BaseModel):
    heart_rate: float
    sdnn: float
    rmssd: float
    eda_mean: float
    eda_std: float
    eda_peaks: float
    arousal_index: float
    resp_mean: float
    resp_std: float
    temp_mean: float
    patient_id: str = Field(default="S2")
    source: str = Field(default="watch_simulation")
    timestamp: Optional[str] = None


class CalibrationStats(BaseModel):
    stats: dict  # {feature_name: {"mean": float, "std": float}}


# ── In-memory state ─────────────────────────────────────────────────────
_buffer_lock = Lock()
_buffer: deque = deque(maxlen=MAX_BUFFER)

_calibration_lock = Lock()
_calibration: dict = {}


# ── Health / status ──────────────────────────────────────────────────────
@app.get("/health")
def health():
    with _buffer_lock:
        n = len(_buffer)
    with _calibration_lock:
        calibrated = bool(_calibration)
    return {"status": "ok", "buffered_readings": n, "calibrated": calibrated}


# ── Ingest (called by watch_simulation.py or a real device) ─────────────
@app.post("/ingest")
def ingest(reading: WatchReading):
    payload = reading.dict()
    payload["timestamp"] = payload["timestamp"] or datetime.now().isoformat()
    with _buffer_lock:
        _buffer.append(payload)
        n = len(_buffer)
    return {"status": "ok", "buffered_readings": n}


# ── Read endpoints (called by app.py's Live Watch tab) ───────────────────
@app.get("/latest")
def latest():
    with _buffer_lock:
        if not _buffer:
            raise HTTPException(status_code=404, detail="No readings yet.")
        return _buffer[-1]


@app.get("/history")
def history(limit: int = 60):
    limit = max(1, min(limit, MAX_BUFFER))
    with _buffer_lock:
        return list(_buffer)[-limit:]


@app.delete("/reset")
def reset():
    with _buffer_lock:
        _buffer.clear()
    return {"status": "ok", "message": "Buffer cleared."}


# ── Calibration (app.py pushes WESAD-derived mean/std after training,
#    so watch_simulation.py can drift around realistic values) ──────────
@app.post("/calibration")
def set_calibration(payload: CalibrationStats):
    missing = [f for f in FEATURES if f not in payload.stats]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing calibration for: {missing}",
        )
    with _calibration_lock:
        _calibration.clear()
        _calibration.update(payload.stats)
    return {"status": "ok", "message": "Calibration stored."}


@app.get("/calibration")
def get_calibration():
    with _calibration_lock:
        if not _calibration:
            raise HTTPException(status_code=404, detail="No calibration set yet.")
        return dict(_calibration)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
