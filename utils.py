"""
Funções de pré-processamento robustas:
- normalização de colunas
- detecção/mapeamento de colunas de falha (aliases + heurística 0/1)
- conversão robusta para 0/1
- imputação, one-hot, escalonamento e split
"""
import pandas as pd
import unicodedata, re
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple, Dict

def _normalize_col_name(col: str) -> str:
    if col is None: return ""
    s = str(col).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s.strip("_")

def carregar_e_processar_dados(file_like) -> pd.DataFrame:
    """Lê CSV e retorna DataFrame com colunas normalizadas."""
    df = pd.read_csv(file_like)
    df.columns = [_normalize_col_name(c) for c in df.columns]
    return df

def _is_binary_series(s: pd.Series) -> bool:
    vals = s.dropna().unique()
    if len(vals) == 0:
        return False
    mapped = set()
    for v in vals:
        if isinstance(v, str):
            v2 = v.strip()
            if v2 in ("0", "1"):
                mapped.add(int(v2))
            else:
                return False
        else:
            try:
                iv = int(v)
                if float(v) != iv:
                    return False
                if iv in (0, 1):
                    mapped.add(iv)
                else:
                    return False
            except Exception:
                return False
    return mapped.issubset({0, 1})

def detect_failure_columns(df_train: pd.DataFrame) -> Dict:
    """
    Tenta mapear as 5 colunas de falha usando aliases e identifica colunas 0/1 candidatas.
    Retorna dict com keys: detected_map (canonical->col ou None), candidate_binary_cols, all_columns
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

    excluded = set(["id", "id_produto", "falha_maquina"])
    candidate_binary_cols = [c for c in cols if c not in excluded and _is_binary_series(df_train[c])]

    return {"detected_map": detected, "candidate_binary_cols": candidate_binary_cols, "all_columns": cols}

def _coerce_to_binary_series(s: pd.Series, col_name: str, verbose: bool=False) -> pd.Series:
    """
    Converte valores da Series para 0/1 com heurísticas:
     - bool -> int
     - numeric: >=0.5 -> 1
     - strings comuns: '1','0','sim','nao','yes','no' -> map
     - se >5% ambíguos -> erro
     - se <=5% ambíguos -> preencher com 0 e avisar (quando verbose)
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    s_nonnull = s.dropna()
    if len(s_nonnull) == 0:
        return pd.Series(0, index=s.index, dtype=int)

    num = pd.to_numeric(s_nonnull, errors='coerce')
    num_ratio = num.notna().sum() / len(s_nonnull)

    if num_ratio >= 0.9:
        full_num = pd.to_numeric(s, errors='coerce').fillna(0)
        return (full_num >= 0.5).astype(int)

    s_str = s.fillna("").astype(str).str.strip().str.lower()
    mapping = {
        '1':1, '0':0, 'true':1, 'false':0, 'sim':1, 's':1, 'nao':0, 'não':0, 'n':0,
        'yes':1, 'no':0, 'y':1, 't':1, 'f':0
    }
    mapped = s_str.map(mapping)

    remaining = mapped.isna()
    if remaining.any():
        num2 = pd.to_numeric(s_str[remaining], errors='coerce')
        mapped.loc[remaining] = (num2 >= 0.5).astype('Int64')

    missing_count = int(mapped.isna().sum())
    total = len(mapped)
    if missing_count > 0:
        missing_ratio = missing_count / total
        sample_bad = pd.Series(s[ mapped.isna() ].unique()).tolist()[:10]
        if missing_ratio > 0.05:
            raise ValueError(
                f"Coluna target '{col_name}' contém {missing_count}/{total} ({missing_ratio:.1%}) valores não mapeáveis. Exemplos: {sample_bad}."
            )
        if verbose:
            print(f"[WARN] Coluna '{col_name}' teve {missing_count} valores ambíguos; preenchendo com 0. Exemplos: {sample_bad}")
        mapped = mapped.fillna(0)

    return mapped.astype(int)

def preprocess_pipeline(
    treino_df: pd.DataFrame,
    teste_df: Optional[pd.DataFrame]=None,
    target_columns: Optional[List[str]]=None,
    verbose: bool=False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], dict, List[str], List[str]]:
    canonical = ["fdf", "fdc", "fp", "fte", "fa"]
    df_train = treino_df.copy()
    df_test = teste_df.copy() if teste_df is not None else None

    if target_columns is not None:
        if not isinstance(target_columns, list) or len(target_columns) != len(canonical):
            raise ValueError("Se passar target_columns, forneça lista com 5 nomes (ordem: fdf, fdc, fp, fte, fa)")
        for c in target_columns:
            if c not in df_train.columns:
                raise KeyError(f"Coluna de alvo fornecida '{c}' não encontrada no treino.")
        detected_actual = target_columns.copy()
    else:
        detection = detect_failure_columns(df_train)
        mapped = detection["detected_map"]
        missing = [k for k, v in mapped.items() if v is None]
        if len(missing) == 0:
            detected_actual = [mapped[k] for k in canonical]
        else:
            candidates = [c for c in detection["candidate_binary_cols"] if c not in mapped.values()]
            if len(candidates) >= len(missing):
                cand_sorted = sorted(candidates, key=lambda c: df_train[c].mean(), reverse=True)
                assigned = {}
                i = 0
                for m in missing:
                    assigned[m] = cand_sorted[i]; i += 1
                detected_actual = []
                for can in canonical:
                    if mapped[can] is not None:
                        detected_actual.append(mapped[can])
                    else:
                        detected_actual.append(assigned[can])
            else:
                raise KeyError(
                    f"Não foi possível mapear automaticamente todos os alvos. Candidatos binários: {detection['candidate_binary_cols']}. Mapeamento parcial: {mapped}."
                )

    if verbose:
        print(f"[INFO] Targets detectados (colunas reais): {detected_actual}")

    drop_cols = ["id", "id_produto", "falha_maquina"]
    features = [c for c in df_train.columns if c not in detected_actual + drop_cols]
    X = df_train[features].copy()

    # converter targets robustamente
    y_dict = {}
    for col in detected_actual:
        try:
            y_dict[col] = _coerce_to_binary_series(df_train[col], col, verbose=verbose)
        except Exception as e:
            raise ValueError(f"Erro ao converter coluna alvo '{col}': {e}")

    y = pd.DataFrame(y_dict)

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    if df_test is not None:
        X_test = df_test[[c for c in df_test.columns if c not in detected_actual + drop_cols]].copy()
        if cat_cols:
            X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)
    else:
        X_test = None

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)

    if X_test is not None:
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test.columns)
    else:
        X_test_scaled = None

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    preprocess_objects = {
        "imputer": imputer,
        "scaler": scaler,
        "target_columns_actual": detected_actual,
        "canonical_targets": canonical
    }

    return X_train, X_val, y_train, y_val, X_test_scaled, preprocess_objects, list(X_train.columns), detected_actual
