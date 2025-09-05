"""
Treinamento, avaliação e geração de predições para submissão.
Compara RandomForest, LightGBM e XGBoost (quando disponíveis).
"""
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score, hamming_loss, accuracy_score
import joblib

# tentar importar LGBM / XGB
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def _evaluate_model(m, X_train, y_train, X_val, y_val, name):
    m.fit(X_train, y_train)
    y_pred = pd.DataFrame(m.predict(X_val), columns=y_val.columns)
    # métricas por label
    per_label = {}
    f1s = []
    for col in y_val.columns:
        p, r, f1, _ = precision_recall_fscore_support(y_val[col], y_pred[col], average='binary', zero_division=0)
        per_label[col] = {"precision": float(p), "recall": float(r), "f1": float(f1)}
        f1s.append(f1)
    mean_f1 = float(np.mean(f1s))
    ham = float(hamming_loss(y_val, y_pred))
    subset_acc = float(accuracy_score(y_val, y_pred))
    return {"model_name": name, "mean_f1": mean_f1, "per_label_f1": per_label, "hamming_loss": ham, "subset_acc": subset_acc, "model": m, "y_val_pred": y_pred}

def comparar_e_treinar_modelos(X_train, y_train, X_val, y_val):
    results = []
    # RandomForest baseline
    rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    results.append(_evaluate_model(rf, X_train, y_train, X_val, y_val, "RandomForest"))

    if LGBM_AVAILABLE:
        lgb = MultiOutputClassifier(LGBMClassifier(n_estimators=500, random_state=42, n_jobs=-1))
        results.append(_evaluate_model(lgb, X_train, y_train, X_val, y_val, "LightGBM"))

    if XGB_AVAILABLE:
        xgb = MultiOutputClassifier(XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1))
        results.append(_evaluate_model(xgb, X_train, y_train, X_val, y_val, "XGBoost"))

    results_sorted = sorted(results, key=lambda r: r["mean_f1"], reverse=True)
    return results_sorted, results_sorted[0]["model"]

def gerar_predicoes_para_submissao(model, X_test_proc, canonical_targets, target_map=None, original_test_df=None):
    """
    Gera DataFrame pronto para submissão:
    - canonical_targets: ['fdf','fdc','fp','fte','fa']
    - target_map: lista com nomes reais usados no treino, na mesma ordem (opcional)
    """
    preds = pd.DataFrame(model.predict(X_test_proc))
    # Se o modelo foi treinado com y colnames (geralmente sim), preds columns podem vir vazias; garantir nomes
    # Se target_map for fornecido: renomear e converter para colunas canônicas
    if target_map is not None:
        preds.columns = target_map
        # mapear para canônicos
        df_sub = pd.DataFrame()
        for canon, real in zip(canonical_targets, target_map):
            if real in preds.columns:
                df_sub[canon] = preds[real].astype(int)
            else:
                df_sub[canon] = 0
    else:
        # sem target_map, assumir que preds tem tamanho correto e nomear com canonical_targets
        preds.columns = canonical_targets[:preds.shape[1]]
        df_sub = preds.astype(int)

    if original_test_df is not None and "id" in original_test_df.columns:
        df_sub.insert(0, "id", original_test_df["id"].values)

    return df_sub
