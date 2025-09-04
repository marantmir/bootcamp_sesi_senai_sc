# app.py
"""
Sistema de Manutenção Preditiva
--------------------------------
Interface em Streamlit para carregar dados, pré-processar
e treinar modelos de machine learning para manutenção preditiva.

Autor: Marco (refatorado com boas práticas)
"""

import streamlit as st
import traceback
import sys
import pandas as pd

from utils import (
    carregar_e_processar_dados,
    preprocessar_dados,
)

# ===============================
# CONFIGURAÇÃO DO APP
# ===============================
st.set_page_config(
    page_title="🔧 Manutenção Preditiva",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Sistema Inteligente de Manutenção Preditiva")
st.markdown("Carregue seus dados de **treino** e (opcionalmente) de **teste** para iniciar.")

# ===============================
# UPLOAD DE ARQUIVOS
# ===============================
arquivo_treino = st.file_uploader("📂 Selecione o arquivo de TREINO (CSV)", type=["csv"])
arquivo_teste = st.file_uploader("📂 Selecione o arquivo de TESTE (CSV) [opcional]", type=["csv"])

if arquivo_treino:
    try:
        treino_df = carregar_e_processar_dados(arquivo_treino)
        teste_df = carregar_e_processar_dados(arquivo_teste) if arquivo_teste else None

        st.success("✅ Arquivo(s) carregado(s) com sucesso!")
        st.write("### Pré-visualização dos dados de treino:")
        st.dataframe(treino_df.head())

        # ===============================
        # PRÉ-PROCESSAMENTO
        # ===============================
        st.subheader("⚙️ Pré-processamento dos Dados")

        X_train, X_test, y_train, y_test, scaler, features = preprocessar_dados(
            treino_df,
            teste_df,
            verbose=True
        )

        st.success("✅ Pré-processamento concluído com sucesso!")
        st.write("**Dimensões:**")
        st.write(f"Treino: {X_train.shape}, Teste: {X_test.shape if not X_test.empty else 'Não fornecido'}")

        st.write("**Colunas utilizadas no modelo:**")
        st.code(features)

        # ===============================
        # (Aqui você pode integrar modelos ML)
        # ===============================
        st.info("📌 Agora é possível treinar modelos de ML com os dados pré-processados.")

    except Exception as e:
        tb = traceback.format_exc()
        st.error("❌ Ocorreu um erro no processamento.")
        st.code(tb)
        print(tb, file=sys.stderr)
