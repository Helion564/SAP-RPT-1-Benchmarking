"""
benchmark.py — Core benchmarking engine for the web app.
Runs XGBoost, LightGBM, CatBoost, and SAP RPT-1 on an uploaded DataFrame
using k-fold cross-validation, then generates a rich model recommendation.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             r2_score, mean_absolute_error, mean_squared_error)
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

warnings.filterwarnings("ignore")

# ── allow importing real model wrappers from ../code ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

N_FOLDS   = int(os.getenv("N_FOLDS",   "5"))
RAND      = int(os.getenv("RANDOM_STATE", "42"))
HF_TOKEN  = os.getenv("HUGGING_FACE_HUB_TOKEN", "")

MODEL_COLORS = {
    "XGBoost":        "#f59e0b",
    "LightGBM":       "#10b981",
    "CatBoost":       "#6366f1",
    "SAP-RPT1":       "#ec4899",
}

# ── Model builders ─────────────────────────────────────────────────────────────

def _xgb(task):
    import xgboost as xgb
    kw = dict(n_estimators=200, max_depth=6, learning_rate=0.1,
               random_state=RAND, verbosity=0, eval_metric="logloss")
    return xgb.XGBClassifier(**kw) if task == "classification" else xgb.XGBRegressor(**kw)

def _lgb(task):
    import lightgbm as lgb
    kw = dict(n_estimators=200, learning_rate=0.1, random_state=RAND, verbose=-1)
    return lgb.LGBMClassifier(**kw) if task == "classification" else lgb.LGBMRegressor(**kw)

def _cat(task):
    from catboost import CatBoostClassifier, CatBoostRegressor
    kw = dict(iterations=200, learning_rate=0.1, random_state=RAND, verbose=False)
    return CatBoostClassifier(**kw) if task == "classification" else CatBoostRegressor(**kw)


class _SAPModel:
    """
    Tries the real SAP RPT-1 OSS via HuggingFace; falls back to k-NN simulator
    if the package is not installed or authentication fails.
    """
    def __init__(self, task):
        self.task = task
        self._real = False
        self._le   = LabelEncoder() if task == "classification" else None

        if HF_TOKEN:
            try:
                from huggingface_hub import login
                login(token=HF_TOKEN, add_to_git_credential=False)
                from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor
                if task == "classification":
                    self._model = SAP_RPT_OSS_Classifier(max_context_size=2048, bagging=1)
                else:
                    self._model = SAP_RPT_OSS_Regressor(max_context_size=2048, bagging=1)
                self._real = True
            except Exception:
                self._init_sim()
        else:
            self._init_sim()

    def _init_sim(self):
        k = 15
        if self.task == "classification":
            self._model = KNeighborsClassifier(n_neighbors=k)
        else:
            self._model = KNeighborsRegressor(n_neighbors=k)

    def fit(self, X, y):
        if self._real:
            self._model.fit(X, y)
        else:
            if self.task == "classification":
                y_enc = self._le.fit_transform(y)
                self._model.fit(X, y_enc)
            else:
                self._model.fit(X, y)
        return self

    def predict(self, X):
        preds = self._model.predict(X)
        if not self._real and self.task == "classification":
            preds = self._le.inverse_transform(preds)
        return preds

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    @property
    def label(self):
        return "SAP-RPT1 (real)" if self._real else "SAP-RPT1 (sim)"


BUILDERS = {
    "XGBoost":  _xgb,
    "LightGBM": _lgb,
    "CatBoost": _cat,
    "SAP-RPT1": lambda task: _SAPModel(task),
}

# ── Preprocessing ──────────────────────────────────────────────────────────────

def _prep(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    num = X.select_dtypes(include=[np.number]).columns
    cat = X.select_dtypes(exclude=[np.number]).columns
    if len(num): X[num] = X[num].fillna(X[num].mean())
    for c in cat:
        X[c] = LabelEncoder().fit_transform(X[c].fillna("__NA__").astype(str))
    return X

def _encode_target(y: pd.Series):
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        return pd.Series(le.fit_transform(y.astype(str)), name=y.name), le
    return y, None

# ── Metrics ───────────────────────────────────────────────────────────────────

def _clf_metrics(model, X_tr, y_tr, X_val, y_val):
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_t = time.perf_counter() - t0
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average="macro", zero_division=0)
    try:
        proba = model.predict_proba(X_val)
        n_cls = len(np.unique(y_val))
        auc = roc_auc_score(y_val, proba[:, 1]) if n_cls == 2 else \
              roc_auc_score(y_val, proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "f1_macro": f1, "roc_auc": auc, "fit_time": fit_t}

def _reg_metrics(model, X_tr, y_tr, X_val, y_val):
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_t = time.perf_counter() - t0
    y_pred = model.predict(X_val)
    return {
        "r2":      r2_score(y_val, y_pred),
        "mae":     mean_absolute_error(y_val, y_pred),
        "rmse":    float(np.sqrt(mean_squared_error(y_val, y_pred))),
        "fit_time": fit_t,
    }

# ── Cross-validation ──────────────────────────────────────────────────────────

def _run_cv(builder, X, y, task):
    if task == "classification":
        splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=RAND).split(X, y))
    else:
        splits = list(KFold(N_FOLDS, shuffle=True, random_state=RAND).split(X))

    fold_results = []
    for tr_idx, val_idx in splits:
        Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[tr_idx], y.iloc[val_idx]
        Xtr, Xval = _prep(Xtr), _prep(Xval)
        model = builder(task)
        if task == "classification":
            fold_results.append(_clf_metrics(model, Xtr, ytr, Xval, yval))
        else:
            fold_results.append(_reg_metrics(model, Xtr, ytr, Xval, yval))

    df = pd.DataFrame(fold_results)
    return {"mean": df.mean().to_dict(), "std": df.std().to_dict(), "folds": df.to_dict("records")}

# ── Recommendation engine ──────────────────────────────────────────────────────

def _recommend(results: dict, task: str) -> dict:
    primary = "roc_auc" if task == "classification" else "r2"
    secondary = "f1_macro" if task == "classification" else "mae"
    higher_secondary = task == "classification"   # True = higher is better

    scores = {}
    for name, data in results.items():
        if "error" in data:
            continue
        m = data["mean"]
        s = data["std"]
        prim_val  = m.get(primary, 0) or 0
        prim_std  = s.get(primary, 1) or 1
        sec_val   = m.get(secondary, 0) or 0
        fit_t     = m.get("fit_time", 99) or 99

        # Normalised composite (0-1 each axis)
        # Primary: 40%, Consistency (1-std): 20%, Speed (1-log-time): 20%, Secondary: 20%
        consistency = max(0.0, 1.0 - prim_std * 10)
        max_t = 60.0
        speed = max(0.0, 1.0 - min(fit_t, max_t) / max_t)
        sec_norm = sec_val if higher_secondary else max(0, 1 - sec_val / (sec_val + 1e-6 + 1))
        composite = 0.40 * prim_val + 0.20 * consistency + 0.20 * speed + 0.20 * sec_norm
        scores[name] = {
            "primary":     round(prim_val, 4),
            "consistency": round(consistency, 4),
            "speed":       round(speed, 4),
            "secondary":   round(sec_val, 4),
            "composite":   round(composite, 4),
            "fit_time":    round(fit_t, 3),
        }

    if not scores:
        return {}

    best_overall  = max(scores, key=lambda n: scores[n]["composite"])
    best_accuracy = max(scores, key=lambda n: scores[n]["primary"])
    best_speed    = max(scores, key=lambda n: scores[n]["speed"])
    best_stable   = max(scores, key=lambda n: scores[n]["consistency"])
    p_metric_label = "ROC-AUC" if task == "classification" else "R²"

    def pct_faster(fast, others):
        fast_t = results[fast]["mean"]["fit_time"]
        other_ts = [results[n]["mean"]["fit_time"] for n in others if n != fast and "error" not in results[n]]
        if not other_ts: return 0
        avg = sum(other_ts) / len(other_ts)
        return round((avg - fast_t) / (avg + 1e-9) * 100, 1)

    recommendations = {
        "best_overall": {
            "model":   best_overall,
            "score":   scores[best_overall]["composite"],
            "reason":  (f"{best_overall} has the highest composite score ({scores[best_overall]['composite']:.4f}), "
                        f"balancing {p_metric_label} ({scores[best_overall]['primary']:.4f}), "
                        f"consistency, and training speed.")
        },
        "best_accuracy": {
            "model":   best_accuracy,
            "score":   scores[best_accuracy]["primary"],
            "reason":  (f"{best_accuracy} achieves the highest {p_metric_label} of "
                        f"{scores[best_accuracy]['primary']:.4f}. Best choice when raw predictive "
                        f"performance is the only priority.")
        },
        "best_speed": {
            "model":   best_speed,
            "score":   scores[best_speed]["fit_time"],
            "reason":  (f"{best_speed} is the fastest model, training in "
                        f"{scores[best_speed]['fit_time']:.3f}s per fold — "
                        f"{pct_faster(best_speed, list(scores.keys()))}% faster than average. "
                        f"Ideal for real-time retraining or large data pipelines.")
        },
        "best_consistency": {
            "model":   best_stable,
            "score":   scores[best_stable]["consistency"],
            "reason":  (f"{best_stable} is the most consistent model across folds, "
                        f"with the lowest variance in {p_metric_label}. "
                        f"Best choice when reliability matters more than peak performance.")
        },
    }

    # Production recommendation: best composite that isn't worst speed
    prod = best_overall
    recommendations["production"] = {
        "model": prod,
        "reason": (f"For production deployment, we recommend {prod}. "
                   f"It achieves an excellent balance of accuracy "
                   f"({scores[prod]['primary']:.4f} {p_metric_label}), "
                   f"trains in {scores[prod]['fit_time']:.3f}s per fold, "
                   f"and performs consistently across data splits.")
    }

    return {"scores": scores, "recommendations": recommendations, "primary_metric": p_metric_label}


# ── Public API ────────────────────────────────────────────────────────────────

def infer_task(y: pd.Series) -> str:
    if y.dtype == object or str(y.dtype) == "category":
        return "classification"
    return "classification" if y.nunique() < 20 else "regression"


def run_benchmark(df: pd.DataFrame, target_col: str) -> dict:
    """
    Run full benchmark on a DataFrame.

    Parameters
    ----------
    df          : the full dataset
    target_col  : name of the target column

    Returns
    -------
    dict with keys: dataset_info, task, results, recommendation
    """
    y_raw = df[target_col].copy()
    X     = df.drop(columns=[target_col]).copy()
    y, _  = _encode_target(y_raw)

    task = infer_task(y_raw)

    dataset_info = {
        "n_samples":  len(df),
        "n_features": X.shape[1],
        "target_col": target_col,
        "task":       task,
        "n_classes":  int(y.nunique()) if task == "classification" else None,
        "columns":    list(X.columns),
    }

    results = {}
    sap_label = None
    for name, builder in BUILDERS.items():
        try:
            cv = _run_cv(builder, X, y, task)
            results[name] = cv
            if name == "SAP-RPT1":
                # capture real/sim label
                try:
                    m = builder(task)
                    sap_label = m.label
                except Exception:
                    sap_label = "SAP-RPT1"
        except Exception as e:
            results[name] = {"error": str(e)}

    if sap_label and "SAP-RPT1" in results and "error" not in results["SAP-RPT1"]:
        results["SAP-RPT1"]["label"] = sap_label

    recommendation = _recommend(results, task)

    return {
        "dataset_info":   dataset_info,
        "task":           task,
        "results":        results,
        "recommendation": recommendation,
        "n_folds":        N_FOLDS,
    }
