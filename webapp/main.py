"""
main.py — FastAPI backend for the SAP RPT-1 Benchmarking Web App.
"""

import io, os
from pathlib import Path
from dotenv import load_dotenv

# Load .env before anything else so HF_TOKEN is available to benchmark.py
load_dotenv(Path(__file__).parent / ".env")

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from benchmark import run_benchmark, infer_task, _INFERENCE_CACHE, prep_for_predict
except ImportError:
    from webapp.benchmark import run_benchmark, infer_task, _INFERENCE_CACHE, prep_for_predict

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "5")) * 1024 * 1024   # default 5 MB

app = FastAPI(title="SAP RPT-1 Benchmarking API", version="1.0.0")

# ── Static files (frontend) ────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

from fastapi.responses import FileResponse

@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── /preview ───────────────────────────────────────────────────────────────────
@app.post("/preview")
async def preview(file: UploadFile = File(...)):
    """
    Return column names + first 5 rows of the uploaded CSV.
    Used by the frontend to let the user pick the target column.
    """
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File too large. Max size is {MAX_FILE_BYTES // (1024*1024)} MB.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if df.shape[1] < 2:
        raise HTTPException(400, "CSV must have at least 2 columns (features + target).")

    # Guess default target: last column
    default_target = df.columns[-1]

    return JSONResponse({
        "columns":        list(df.columns),
        "default_target": default_target,
        "n_rows":         len(df),
        "n_cols":         df.shape[1],
        "preview":        df.head(5).fillna("").to_dict("records"),
    })


# ── /benchmark ─────────────────────────────────────────────────────────────────
@app.post("/benchmark")
async def benchmark(
    file:       UploadFile = File(...),
    target_col: str        = Form(...),
):
    """
    Upload a CSV + specify target column → get full benchmark results.
    """
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File too large. Max {MAX_FILE_BYTES // (1024*1024)} MB.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if target_col not in df.columns:
        raise HTTPException(400, f"Column '{target_col}' not found. Available: {list(df.columns)}")

    if df.shape[0] < 20:
        raise HTTPException(400, "Dataset too small (need at least 20 rows).")

    try:
        result = run_benchmark(df, target_col)
    except Exception as e:
        raise HTTPException(500, f"Benchmarking failed: {e}")

    return JSONResponse(result)


# ── /predict_single ────────────────────────────────────────────────────────────
@app.post("/predict_single")
async def predict_single(row: dict):
    """
    Accepts a JSON dict of feature values.
    Returns live prediction from the best model trained during the last benchmark run.
    """
    cache = _INFERENCE_CACHE
    if cache["model"] is None:
        raise HTTPException(400, "No model trained yet. Run benchmarking first.")

    try:
        df = prep_for_predict(row, cache["columns"])
        model = cache["model"]
        task  = cache["task"]
        le    = cache["le"]

        if task == "classification":
            pred_int   = int(model.predict(df)[0])
            pred_label = le.inverse_transform([pred_int])[0] if le else str(pred_int)
            try:
                proba   = model.predict_proba(df)[0]
                classes = list(le.classes_) if le else [str(i) for i in range(len(proba))]
                probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
                confidence    = float(max(proba))
            except Exception:
                probabilities = {}
                confidence    = None
            return JSONResponse({
                "prediction":    str(pred_label),
                "confidence":    confidence,
                "probabilities": probabilities,
                "task":          "classification",
            })
        else:
            pred = float(model.predict(df)[0])
            return JSONResponse({"prediction": pred, "task": "regression"})

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
