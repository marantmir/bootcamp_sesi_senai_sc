import pandas as pd
import unicodedata, re
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def _normalize_col_name(col: str) -> str:
    if col is None: return ""
    s = str(col).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("utf-8")
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"__+", "_", s).strip("_")
    return s

def carregar_e_processar_dados(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    df.columns = [_normalize_col_name(c) for c in df.columns]
    return df

def preprocess_pipeline(treino_df, teste_df=None, targets=None, verbose=False):
    # targets padrão
    if targets is None:
        targets = ["fdf", "fdc", "fp", "fte", "fa"]

    df_train = treino_df.copy()
    df_test = teste_df.copy() if teste_df is not None else None

    # checar targets
    missing = [t for t in targets if t not in df_train.columns]
    if missing:
        raise KeyError(f"Rótulos faltando no treino: {missing}")

    drop_cols = ["id", "id_produto", "falha_maquina"]
    features = [c for c in df_train.columns if c not in targets + drop_cols]

    X = df_train[features].copy()
    y = df_train[targets].astype(int).copy()

    # one-hot em categoricas
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # processar teste
    if df_test is not None:
        X_test = df_test[[c for c in df_test.columns if c not in targets + drop_cols]].copy()
        if cat_cols:
            X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        # alinhar colunas
        X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
    else:
        X_test = None

    # imputer e scaler
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
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=None)

    return X_train, X_val, y_train, y_val, X_test_scaled, {"imputer": imputer, "scaler": scaler}, list(X_train.columns), targets
