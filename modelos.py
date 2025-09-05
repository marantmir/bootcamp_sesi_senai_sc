import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

def treinar_modelo(X_train, y_train):
    """Treina um modelo multirrótulo usando RandomForest."""
    base_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)
    return model

def gerar_predicoes(modelo, X_test, targets):
    """Gera predições multirrótulo no formato exigido pela API."""
    y_pred = modelo.predict(X_test)
    predicoes = pd.DataFrame(y_pred, columns=targets)
    return predicoes
