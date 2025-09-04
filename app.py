import streamlit as st
import pandas as pd
from utils import (
    carregar_dados,
    preprocessar_dados,
    treinar_modelos,
    avaliar_modelos,
    exibir_resultados
)

# 🎯 Título
st.title("🔧 Projeto de Manutenção Preditiva")
st.write("Comparação de modelos de Machine Learning para prever falhas.")

# 📂 Upload de arquivos
st.sidebar.header("📂 Upload dos Dados")
arquivo_treino = st.sidebar.file_uploader("Selecione o arquivo de treino", type=["csv"])
arquivo_teste = st.sidebar.file_uploader("Selecione o arquivo de teste", type=["csv"])

if arquivo_treino and arquivo_teste:
    # 1. Carregar dados
    treino, teste = carregar_dados(arquivo_treino, arquivo_teste)

    # 2. Pré-processamento
    X_train, X_test, y_train, y_test, scaler, features_cols = preprocessar_dados(treino, teste, verbose=True)

    # 3. Treinar modelos
    modelos, historico = treinar_modelos(X_train, y_train)

    # 4. Avaliar modelos
    resultados = avaliar_modelos(modelos, X_test, y_test)

    # 5. Exibir resultados
    exibir_resultados(resultados, historico)

else:
    st.warning("Por favor, faça o upload dos arquivos de treino e teste.")

