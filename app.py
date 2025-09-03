import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ---------------- Funções Auxiliares ----------------
def treinar_modelo_multirrotulo(X_train, y_train, X_valid, y_valid, colunas_falha, n_estimators):
    modelo = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42, n_jobs=-1
    )
    modelo.fit(X_train, y_train)
    pred_valid = modelo.predict(X_valid)

    metricas_classes = {}
    for i, col in enumerate(colunas_falha):
        acc = accuracy_score(y_valid[:, i], pred_valid[:, i])
        metricas_classes[col] = acc

    return modelo, pred_valid, metricas_classes

# ---------------- Upload de Dados ----------------
st.title("⚙️ Monitoramento e Predição de Falhas em Máquinas")

arquivo = st.file_uploader("📂 Carregue o dataset (CSV)", type="csv")

if arquivo:
    dados_treino = pd.read_csv(arquivo)

    # ===============================
    # 📊 Análise Exploratória
    # ===============================
    st.header("🔎 Análise Exploratória de Dados")

    st.subheader("📊 Distribuição da Variável Alvo")
    if 'falha_maquina' in dados_treino.columns:
        fig, ax = plt.subplots()
        dados_treino['falha_maquina'].value_counts().plot(
            kind='bar', ax=ax, color='skyblue'
        )
        ax.set_title("Distribuição de Falhas")
        ax.set_ylabel("Frequência")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Coluna 'falha_maquina' não encontrada. Verifique se o dataset está correto.")

    st.subheader("📉 Correlação entre Variáveis Numéricas")
    corr = dados_treino.corr(numeric_only=True)
    if not corr.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

    if 'falha_maquina' in dados_treino.columns:
        st.subheader("🔎 Variáveis mais correlacionadas com Falha")
        corr_target = corr['falha_maquina'].drop('falha_maquina').sort_values(ascending=False)
        st.bar_chart(corr_target.head(10))

    st.subheader("🧹 Valores Ausentes (%)")
    ausentes = dados_treino.isnull().mean().sort_values(ascending=False) * 100
    st.bar_chart(ausentes.head(10))

    # ===============================
    # ⚙️ Treinamento do Modelo
    # ===============================
    st.header("🤖 Treinamento do Modelo")

    tipo_modelagem = st.selectbox(
        "Selecione o tipo de modelagem",
        ["Binária", "Multiclasse", "Multirrótulo"]
    )

    percentual_teste = st.slider("Percentual de dados para teste", 10, 50, 20)
    n_estimators = st.slider("Número de Árvores (RandomForest)", 10, 500, 100, step=10)
    semente_aleatoria = st.number_input("Semente Aleatória", 0, 9999, 42)

    # ---------------- Definir Target ----------------
    tipo_target = None
    y = None

    if tipo_modelagem == "Multirrótulo":
        colunas_falha = [col for col in dados_treino.columns if "falha" in col.lower()]
        if not colunas_falha:
            st.error("❌ Dados não contêm colunas de falha necessárias para modelagem multirrótulo")
            st.stop()
        y = dados_treino[colunas_falha].values
        tipo_target = "multirrotulo"

    elif tipo_modelagem == "Binária":
        if "falha_maquina" not in dados_treino.columns:
            st.error("❌ Coluna 'falha_maquina' não encontrada para modelagem binária")
            st.stop()
        y = pd.to_numeric(dados_treino["falha_maquina"], errors="coerce")
        if y.nunique() < 2:
            st.error("❌ Dados não contêm informações de falha necessárias para modelagem binária")
            st.stop()
        tipo_target = "binario"

    else:  # Multiclasse
        if "falha_maquina" not in dados_treino.columns:
            st.error("❌ Coluna 'falha_maquina' não encontrada para modelagem multiclasse")
            st.stop()
        y = dados_treino["falha_maquina"].astype(str)
        if y.nunique() < 2:
            st.error("❌ Dados não contêm informações de falha necessárias para modelagem multiclasse")
            st.stop()
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        tipo_target = "multiclasse"

    # ---------------- Preparar Features ----------------
    X = dados_treino.drop(columns=[col for col in dados_treino.columns if "falha" in col.lower()])
    X = pd.get_dummies(X, drop_first=True)

    # ---------------- Split ----------------
    if tipo_target == "multirrotulo":
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=percentual_teste/100, random_state=semente_aleatoria
        )
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=percentual_teste/100, random_state=semente_aleatoria,
            stratify=y if len(np.unique(y)) > 1 else None
        )

    # ---------------- Treinamento ----------------
    with st.spinner("🔄 Treinando modelo..."):
        if tipo_target == "multirrotulo":
            modelo, pred_valid, metricas_classes = treinar_modelo_multirrotulo(
                X_train, y_train, X_valid, y_valid, colunas_falha, n_estimators
            )

            st.subheader("📈 Métricas por Classe")
            cols = st.columns(len(colunas_falha))
            for i, (classe, acc) in enumerate(metricas_classes.items()):
                with cols[i]:
                    st.metric(classe, f"{acc:.3f}")

            st.metric("Média Macro", f"{np.mean(list(metricas_classes.values())):.3f}")

        else:
            modelo = RandomForestClassifier(
                n_estimators=n_estimators, random_state=semente_aleatoria, n_jobs=-1
            )
            modelo.fit(X_train, y_train)
            pred_valid = modelo.predict(X_valid)

            acc = accuracy_score(y_valid, pred_valid)
            st.metric("Acurácia de Validação", f"{acc:.4f}")

            # Relatório detalhado
            report = classification_report(y_valid, pred_valid, output_dict=True, zero_division=0)
            df_report = pd.DataFrame(report).transpose()
            st.subheader("📋 Relatório Detalhado")
            st.dataframe(df_report.style.background_gradient(cmap="Blues"), use_container_width=True)

            # Matriz de confusão interativa
            cm = confusion_matrix(y_valid, pred_valid)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Matriz de Confusão",
                x=np.unique(y_valid),
                y=np.unique(y_valid),
                color_continuous_scale="Blues"
            )
            fig_cm.update_xaxes(title="Predito")
            fig_cm.update_yaxes(title="Real")
            st.plotly_chart(fig_cm, use_container_width=True)
