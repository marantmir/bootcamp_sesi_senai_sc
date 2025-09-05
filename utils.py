import pandas as pd
import unicodedata, re
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple

# ===============================
# FUNÇÕES AUXILIARES
# ===============================
def _normalize_col_name(col: str) -> str:
    if col is None: return ""
    s = str(col).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"__+", "_", s).strip("_")
    return s

def carregar_e_processar_dados(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    df.columns = [_normalize_col_name(c) for c in df.columns]
    return df

# ===============================
# PRÉ-PROCESSAMENTO
# ===============================
def preprocess_pipeline(
    treino_df: pd.DataFrame,
    teste_df: Optional[pd.DataFrame] = None,
    targets: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], dict, List[str], List[str]]:

    # Aliases possíveis para cada alvo
    alias_targets = {
        "fdf": ["fdf", "falha_desgaste_ferramenta", "falhadesgasteferramenta"],
        "fdc": ["fdc", "falha_dissipacao_calor", "falhadissipacaocalor"],
        "fp":  ["fp", "falha_potencia"],
        "fte": ["fte", "falha_tensao_excessiva", "falhatensaoexcessiva"],
        "fa":  ["fa", "falha_aleatoria"]
    }

    df_train = treino_df.copy()
    df_test = teste_df.copy() if teste_df is not None else None

    # Detectar targets automaticamente
    detected_targets = []
    for key, aliases in alias_targets.items():
        found = next((col for col in df_train.columns if col in aliases), None)
        if found:
            detected_targets.append(found)
        else:
            raise KeyError(f"Não encontrei nenhuma coluna correspondente ao alvo '{key}'. Verifique os nomes no CSV.")

    if verbose:
        print(f"[INFO] Targets detectados: {detected_targets}")

    drop_cols = ["id", "id_produto", "falha_maquina"]
    features = [c for c in df_train.columns if c not in detected_targets + drop_cols]

    X = df_train[features].copy()
    y = df_train[detected_targets].astype(int).copy()

    # One-hot em variáveis categóricas
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Processar teste
    if df_test is not None:
        X_test = df_test[[c for c in df_test.columns if c not in detected_targets + drop_cols]].copy()
        if cat_cols:
            X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)
    else:
        X_test = None

    # Imputer + scaler
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)

    if X_test is not None:
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test.columns)
    else:
        X_test_scaled = None

    # Split treino/validação
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return (
        X_train, X_val, y_train, y_val,
        X_test_scaled,
        {"imputer": imputer, "scaler": scaler},
        list(X_train.columns),
        detected_targets
    )
