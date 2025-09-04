# utils.py
import re
import unicodedata
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def _normalize_col_name(col: str) -> str:
    """Normaliza nomes de colunas: strip, lower, remove acentos, substitui não-alphanum por '_'."""
    if col is None:
        return ""
    s = str(col).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", errors="ignore").decode("utf-8")
    s = re.sub(r"[^\w]+", "_", s)          # tudo que não for letra/dígito/_ -> _
    s = re.sub(r"__+", "_", s)             # colapsa underscores duplos
    s = s.strip("_")
    return s

def preprocessar_dados(
    treino: pd.DataFrame,
    teste: Optional[pd.DataFrame] = None,
    possiveis_alvos: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Optional[np.ndarray], StandardScaler, List[str]]:
    """
    Pré-processa treino e teste de forma robusta:
    - normaliza nomes de colunas
    - detecta coluna alvo (com fallback)
    - alinha features entre treino/teste (get_dummies em concat)
    - imputação por mediana (usando apenas treino)
    - StandardScaler (fit no treino)
    - label encoding seguro para y
    Retorna: X_train_df, X_test_df, y_train, y_test (ou None se não existir no teste), scaler, features_cols
    """
    # cópias defensivas
    treino = treino.copy()
    teste = teste.copy() if teste is not None else pd.DataFrame()

    # normalização das colunas (criar mapping original->normalizado)
    train_map = {c: _normalize_col_name(c) for c in treino.columns}
    treino.rename(columns=train_map, inplace=True)

    if not teste.empty:
        test_map = {c: _normalize_col_name(c) for c in teste.columns}
        teste.rename(columns=test_map, inplace=True)

    if possiveis_alvos is None:
        possiveis_alvos = ["target", "label", "classe", "y", "falha", "fdf", "fdc", "fp", "fte", "fa"]

    possiveis_alvos_norm = [_normalize_col_name(x) for x in possiveis_alvos]

    # DETECTAR coluna alvo no treino (preferencialmente)
    coluna_alvo = None
    for col in treino.columns:
        if col in possiveis_alvos_norm:
            coluna_alvo = col
            break

    if coluna_alvo is None:
        # se nada bateu, usar última coluna do treino
        coluna_alvo = treino.columns[-1]
        if verbose:
            print(f"[preprocessar_dados] ⚠️ Coluna alvo não encontrada explicitamente. Usando '{coluna_alvo}' (última coluna) como target.")

    # separar X/y no treino
    y_train_raw = treino[coluna_alvo]
    X_train_raw = treino.drop(columns=[coluna_alvo])

    # preparar X_test_raw e y_test_raw (se existir coluna alvo no teste)
    if not teste.empty and coluna_alvo in teste.columns:
        y_test_raw = teste[coluna_alvo]
        X_test_raw = teste.drop(columns=[coluna_alvo])
    else:
        y_test_raw = None
        X_test_raw = teste if not teste.empty else pd.DataFrame()

    # Alinhar features: concat para dummies consistentes
    # Se X_test_raw vazio, apenas transforme X_train_raw
    if X_test_raw.empty:
        concat = X_train_raw.reset_index(drop=True)
        split_index = len(X_train_raw)
    else:
        concat = pd.concat([X_train_raw.reset_index(drop=True), X_test_raw.reset_index(drop=True)], axis=0, ignore_index=True, sort=False)
        split_index = len(X_train_raw)

    # Aplicar get_dummies para categoricas (garante colunas consistentes)
    concat_d = pd.get_dummies(concat, dummy_na=False)

    # split novamente
    X_train_d = concat_d.iloc[:split_index, :].copy().reset_index(drop=True)
    X_test_d = concat_d.iloc[split_index:, :].copy().reset_index(drop=True) if split_index < len(concat_d) else pd.DataFrame(columns=concat_d.columns)

    # Imputação: usar mediana do treino
    medians = X_train_d.median()
    X_train_d = X_train_d.fillna(medians)
    if not X_test_d.empty:
        X_test_d = X_test_d.fillna(medians)

    # Escalonamento
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_d), columns=X_train_d.columns, index=X_train_d.index)
    if not X_test_d.empty:
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_d), columns=X_test_d.columns, index=X_test_d.index)
    else:
        X_test_scaled = pd.DataFrame(columns=X_train_d.columns)

    # Label encoding seguro para y
    y_train_enc = None
    y_test_enc = None
    if y_train_raw is not None:
        # transformar séries para string quando for object, senão manter
        if pd.api.types.is_object_dtype(y_train_raw) or pd.api.types.is_string_dtype(y_train_raw):
            encoder = LabelEncoder()
            y_train_enc = encoder.fit_transform(y_train_raw.astype(str))
            if y_test_raw is not None:
                # mapear valores desconhecidos para -1
                y_test_list = []
                for v in y_test_raw.astype(str):
                    try:
                        y_test_list.append(int(encoder.transform([v])[0]))
                    except ValueError:
                        # valor novo no teste — marcar como -1
                        y_test_list.append(-1)
                y_test_enc = np.array(y_test_list)
        else:
            # já numérico
            y_train_enc = y_train_raw.values
            if y_test_raw is not None:
                y_test_enc = y_test_raw.values

    # features list
    features_cols = list(X_train_scaled.columns)

    if verbose:
        print(f"[preprocessar_dados] Shapes -> X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}, y_train: {None if y_train_enc is None else y_train_enc.shape}, y_test: {None if y_test_enc is None else y_test_enc.shape}")
        print(f"[preprocessar_dados] Coluna alvo detectada (normalizada): '{coluna_alvo}'")
        if not X_test_scaled.empty:
            print(f"[preprocessar_dados] Colunas comuns após dummies: {len(features_cols)}")

    return X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, scaler, features_cols
