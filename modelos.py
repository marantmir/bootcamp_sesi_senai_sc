# modelos.py
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
import joblib

# tenta importar LightGBM / XGBoost; se não houver, pula
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except:
    LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

def _eval_model(m, X_train, y_train, X_val, y_val, name):
    m.fit(X_train, y_train)
    y_pred = pd.DataFrame(m.predict(X_val), columns=y_val.columns)
    f1s = [f1_score(y_val[col], y_pred[col], zero_division=0) for col in y_val.columns]
    return {
        "model_name": name,
        "mean_f1": float(np.mean(f1s)),
        "per_label_f1": dict(zip(y_val.columns, f1s)),
        "hamming_loss": float(hamming_loss(y_val, y_pred)),
        "subset_acc": float(accuracy_score(y_val, y_pred)),
        "model": m,
        "y_val_pred": y_pred
    }

def comparar_e_treinar_modelos(X_train, y_train, X_val, y_val):
    results = []
    rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    results.append(_eval_model(rf, X_train, y_train, X_val, y_val, "RandomForest"))

    if LGBM_AVAILABLE:
        lgb = MultiOutputClassifier(LGBMClassifier(n_estimators=500, random_state=42, n_jobs=-1))
        results.append(_eval_model(lgb, X_train, y_train, X_val, y_val, "LightGBM"))

    if XGB_AVAILABLE:
        xgb = MultiOutputClassifier(XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1))
        results.append(_eval_model(xgb, X_train, y_train, X_val, y_val, "XGBoost"))

    # ordenar por mean_f1 desc e retornar melhor
    results_sorted = sorted(results, key=lambda r: r["mean_f1"], reverse=True)
    best = results_sorted[0]["model"]
    return results_sorted, best

def gerar_predicoes_para_submissao(model, X_test_proc, targets, original_test_df=None):
    preds = pd.DataFrame(model.predict(X_test_proc), columns=targets)
    # garantir inteiros 0/1
    preds = preds.astype(int)
    if original_test_df is not None and "id" in original_test_df.columns:
        preds.insert(0, "id", original_test_df["id"].values)
    return preds
