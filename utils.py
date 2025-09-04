"""
Módulo utilitário para pré-processamento de dados.
Refatorado com boas práticas de engenharia de dados.
"""

import re
import unicodedata
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ===============================
# FUNÇÕES AUXILIARES
# ===============================
def _normalize_col_name(col: str) -> str:
    """Normaliza nomes de colunas (minúsculas, sem acentos, snake_case)."""
    if col is None:
        return ""
    s = str(col).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", errors="ignore").decode("utf-8")
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s.strip("_")


# ===============================
# LEITURA DE DADOS
# ===============================
def carregar_e_processar_dados(arquivo) -> pd.DataFrame:
    """Carrega CSV e normaliza nomes das colunas."""
    df = pd.read_csv(arquivo)
    mapping = {c: _normalize_col_name(c) for c in df.columns}
    df.rename(columns=mapping, inplace=True)
    return df


# ===============================
# PRÉ-PROCESSAMENTO
# ===============================
def preprocessar_dados(
    treino: pd.DataFrame,
    teste: Optional[pd.DataFrame] = None,
    possiveis_alvos: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Optional[np.ndarray], StandardScaler, List[str]]:
    """
    - Detecta coluna alvo automaticamente
    - Alinha colunas entre treino e teste
    - Imputa valores ausentes pela mediana
    - Aplica StandardScaler
    - Codifica labels

    Retorna:
    (X_train, X_test, y_train, y_test, scaler, features)
    """
    treino = treino.copy()
    teste = teste.copy() if teste is not None else pd.DataFrame()

    # possíveis nomes da coluna alvo
    if possiveis_alvos is None:
        possiveis_alvos = ["target", "label", "classe", "y", "falha", "fdf", "fdc", "fp", "fte", "fa"]

    possiveis_alvos_norm = [_normalize_col_name(x) for x in possiveis_alvos]

    # identificar coluna alvo
    coluna_alvo = next((col for col in treino.columns if col in possiveis_alvos_norm), None)
    if coluna_alvo is None:
        coluna_alvo = treino.columns[-1]  # fallback
        if verbose:
            print(f"[INFO] Nenhuma coluna alvo conhecida encontrada. Usando '{coluna_alvo}'.")

    # separar treino
    y_train_raw = treino[coluna_alvo]
    X_train_raw = treino.drop(columns=[coluna_alvo])

    # separar teste (se existir)
    y_test_raw = None
    X_test_raw = pd.DataFrame()
    if not teste.empty:
        if coluna_alvo in teste.columns:
            y_test_raw = teste[coluna_alvo]
            X_test_raw = teste.drop(columns=[coluna_alvo])
        else:
            X_test_raw = teste.copy()

    # alinhar categorias
    concat = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True, sort=False)
    concat_d = pd.get_dummies(concat)

    split_index = len(X_train_raw)
    X_train_d = concat_d.iloc[:split_index, :].reset_index(drop=True)
    X_test_d = concat_d.iloc[split_index:, :].reset_index(drop=True) if not X_test_raw.empty else pd.DataFrame(columns=concat_d.columns)

    # imputação
    medians = X_train_d.median()
    X_train_d.fillna(medians, inplace=True)
    if not X_test_d.empty:
        X_test_d.fillna(medians, inplace=True)

    # escalonamento
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_d), columns=X_train_d.columns)
    X_test = pd.DataFrame(scaler.transform(X_test_d), columns=X_test_d.columns) if not X_test_d.empty else pd.DataFrame(columns=X_train_d.columns)

    # label encoding
    y_train, y_test = None, None
    if y_train_raw is not None:
        if pd.api.types.is_object_dtype(y_train_raw):
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train_raw.astype(str))
            if y_test_raw is not None:
                y_test = [
                    encoder.transform([v])[0] if v in encoder.classes_ else -1
                    for v in y_test_raw.astype(str)
                ]
        else:
            y_train = y_train_raw.values
            y_test = y_test_raw.values if y_test_raw is not None else None

    if verbose:
        print(f"[INFO] X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"[INFO] Coluna alvo: {coluna_alvo}")

    return X_train, X_test, y_train, y_test, scaler, list(X_train.columns)
