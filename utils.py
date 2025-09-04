import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# =====================
# FUNÇÕES DE SUPORTE
# =====================

def carregar_dados(arquivo_treino, arquivo_teste):
    """Carrega os dados de treino e teste a partir de arquivos CSV."""
    treino = pd.read_csv(arquivo_treino)
    teste = pd.read_csv(arquivo_teste)
    return treino, teste


def preprocessar_dados(treino, teste):
    """Pré-processa os dados: normalização e codificação."""
    X_train = treino.drop("target", axis=1)
    y_train = treino["target"]

    X_test = teste.drop("target", axis=1)
    y_test = teste["target"]

    # Normalização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Garantir que rótulos sejam numéricos
    if y_train.dtype == "object":
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

    return X_train, X_test, y_train, y_test


def treinar_modelos(X_train, y_train):
    """Treina os modelos e retorna os objetos treinados e histórico."""
    modelos = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    }

    historico = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        historico[nome] = modelo
    return modelos, historico


def avaliar_modelos(modelos, X_test, y_test):
    """Avalia os modelos e retorna os resultados de acurácia e relatórios."""
    resultados = {}
    for nome, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        matriz = confusion_matrix(y_test, y_pred)
        resultados[nome] = {"acuracia": acc, "relatorio": report, "matriz": matriz}
    return resultados


def exibir_resultados(resultados, historico):
    """Exibe os resultados na interface Streamlit."""
    st.subheader("📊 Resultados dos Modelos")
    for nome, resultado in resultados.items():
        st.write(f"### 🔹 {nome}")
        st.write(f"**Acurácia:** {resultado['acuracia']:.2f}")
        st.json(resultado["relatorio"])

        # Plot da matriz de confusão
        fig, ax = plt.subplots()
        sns.heatmap(resultado["matriz"], annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Matriz de Confusão - {nome}")
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        st.pyplot(fig)
