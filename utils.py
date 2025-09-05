import pandas as pd
import unicodedata, re
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple, Dict

# -------------------
# Normalização de nomes
# -------------------
def _normalize_col_name(col: str) -> str:
    if col is None:
        return ""
    s = str(col).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s.strip("_")

def carregar_e_processar_dados(file_like) -> pd.DataFrame:
    """Lê CSV e normaliza nomes de colunas."""
    df = pd.read_csv(file_like)
    df.columns = [_normalize_col_name(c) for c in df.columns]
    return df

# -------------------
# Heurística para detectar colunas binárias (0/1)
# -------------------
def _is_binary_series(s: pd.Series) -> bool:
    vals = s.dropna().unique()
    if len(vals) == 0:
        return False
    # Normalize values to ints 0/1 if possible
    mapped = set()
    for v in vals:
        # strings "0"/"1"
        if isinstance(v, str):
            v2 = v.strip()
            if v2 in ("0", "1"):
                mapped.add(int(v2))
            else:
                return False
        else:
            try:
                iv = int(v)
                # allow floats that are integral 0/1 as well
                if float(v) != iv:
                    return False
                if iv in (0, 1):
                    mapped.add(iv)
                else:
                    return False
            except Exception:
                return False
    return mapped.issubset({0, 1})

# -------------------
# Função para detectar colunas de falha (aliases + heurística binária)
# -------------------
def detect_failure_columns(df_train: pd.DataFrame) -> Dict:
    """
    Tenta mapear automaticamente os 5 alvos (fdf, fdc, fp, fte, fa).
    Retorna um dict com:
      - detected_map: {canonical: coluna_encontrada_or_None}
      - candidate_binary_cols: lista de colunas binárias 0/1 (candidatas)
      - all_columns: lista completa de colunas no df
    """
    canonical = ["fdf", "fdc", "fp", "fte", "fa"]
    alias_targets = {
        "fdf": ["fdf", "falha_desgaste_ferramenta", "falhadesgasteferramenta", "f_df"],
        "fdc": ["fdc", "falha_dissipacao_calor", "falhadissipacaocalor", "f_dc"],
        "fp":  ["fp", "falha_potencia", "falhapotencia"],
        "fte": ["fte", "falha_tensao_excessiva", "falhatensaoexcessiva"],
        "fa":  ["fa", "falha_aleatoria", "falha_aleatória"]
    }

    cols = list(df_train.columns)
    detected = {}
    for can in canonical:
        found = next((c for c in cols if c in alias_targets.get(can, [])), None)
        detected[can] = found

    # heurística: detectar colunas 0/1
    excluded = set(["id", "id_produto", "falha_maquina"])
    candidate_binary_cols = [c for c in cols if c not in excluded and _is_binary_series(df_train[c])]

    return {
        "detected_map": detected,
        "candidate_binary_cols": candidate_binary_cols,
        "all_columns": cols
    }

# -------------------
# Pipeline principal (aceita target_columns manual opcional)
# -------------------
def preprocess_pipeline(
    treino_df: pd.DataFrame,
    teste_df: Optional[pd.DataFrame] = None,
    target_columns: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], dict, List[str], List[str]]:
    """
    - Se target_columns for fornecido, usa esses nomes diretamente (ordem: [fdf, fdc, fp, fte, fa]).
    - Caso contrário, tenta mapear automaticamente com detect_failure_columns.
    Retorna:
      X_train, X_val, y_train, y_val, X_test_scaled, preprocess_objects, feature_names, detected_targets_actual
    """
    canonical = ["fdf", "fdc", "fp", "fte", "fa"]
    df_train = treino_df.copy()
    df_test = teste_df.copy() if teste_df is not None else None

    if target_columns is not None:
        # target_columns pode ser lista de nomes reais (já normalizados) na ordem canônica
        if not isinstance(target_columns, list) or len(target_columns) != len(canonical):
            raise ValueError("Se passar target_columns, forneça lista com 5 nomes na ordem: fdf, fdc, fp, fte, fa")
        for c in target_columns:
            if c not in df_train.columns:
                raise KeyError(f"Coluna de alvo fornecida '{c}' não encontrada no treino.")
        detected_actual = target_columns.copy()
    else:
        detection = detect_failure_columns(df_train)
        mapped = detection["detected_map"]
        missing = [k for k, v in mapped.items() if v is None]

        if len(missing) == 0:
            # todos mapeados
            detected_actual = [mapped[k] for k in canonical]
        else:
            # preencha faltantes com candidatas binárias (caso existam)
            candidates = [c for c in detection["candidate_binary_cols"] if c not in mapped.values()]
            if len(candidates) >= len(missing):
                # atribuir candidatos aos missing (ordem por média decrescente para priorizar colunas com mais ocorrências)
                cand_sorted = sorted(candidates, key=lambda c: df_train[c].mean(), reverse=True)
                assigned = {}
                i = 0
                for m in missing:
                    assigned[m] = cand_sorted[i]
                    i += 1
                # construir lista final na ordem canônica
                detected_actual = []
                for can in canonical:
                    if mapped[can] is not None:
                        detected_actual.append(mapped[can])
                    else:
                        detected_actual.append(assigned[can])
            else:
                # não conseguiu mapear automaticamente todos
                raise KeyError(
                    f"Não foi possível mapear automaticamente todos os alvos. "
                    f"Encontrei candidatos binários: {detection['candidate_binary_cols']}. "
                    f"Mapeamento parcial: {mapped}. "
                    "Você pode chamar preprocess_pipeline passando explicitamente `target_columns` (lista com os 5 nomes normalizados) "
                    "ou usar a UI para mapear manualmente."
                )

    if verbose:
        print(f"[INFO] Targets finais (colunas reais no dataset): {detected_actual}")

    drop_cols = ["id", "id_produto", "falha_maquina"]
    features = [c for c in df_train.columns if c not in detected_actual + drop_cols]

    X = df_train[features].copy()
    y = df_train[detected_actual].astype(int).copy()

    # one-hot para categóricas
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # preparar teste
    if df_test is not None:
        X_test = df_test[[c for c in df_test.columns if c not in detected_actual + drop_cols]].copy()
        if cat_cols:
            X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)
    else:
        X_test = None

    # imputação + escalonamento
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)

    if X_test is not None:
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test.columns)
    else:
        X_test_scaled = None

    # split treino/val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    preprocess_objects = {
        "imputer": imputer,
        "scaler": scaler,
        "target_columns_actual": detected_actual,
        "canonical_targets": canonical
    }

    return X_train, X_val, y_train, y_val, X_test_scaled, preprocess_objects, list(X_train.columns), detected_actual
