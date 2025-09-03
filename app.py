# app.py
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from utils import (
    carregar_dados,
    preparar_dados,
    plotar_histograma,
    plotar_correlacao,
    plotar_matriz_confusao,
    plotar_importancia_variaveis,
    montar_payload_api
)

# ---------------------------
# Configuração da página
# ---------------------------
st.set_page_config(page_title="Sistema de Manutenção Preditiva", page_icon="🔧", layout="wide")
st.title("🔧 Sistema Inteligente de Manutenção Preditiva para Indústria 4.0")
st.markdown("Painel para explorar dados, treinar modelo e avaliar previsões via API externa.")
st.markdown("---")

# ---------------------------
# Sidebar — Configurações
# ---------------------------
with st.sidebar:
    st.header("⚙️ Configurações")
    st.subheader("📁 Fonte dos Dados")
    arquivo_treino = st.file_uploader("Selecione o bootcamp_train.csv", type=["csv"], key="up_train")
    arquivo_teste = st.file_uploader("Selecione o bootcamp_test.csv (opcional)", type=["csv"], key="up_test")

    st.markdown("---")
    st.subheader("🤖 Modelagem")
    tipo_modelagem = st.selectbox("Tipo de Modelagem:", ["Binária (Qualquer Falha)", "Multiclasse (Tipos de Falha)"])
    percentual_teste = st.slider("Percentual de Teste (%)", 10, 40, 20, step=5)
    semente_aleatoria = st.number_input("Semente Aleatória", value=42, min_value=0, max_value=9999, step=1)

    st.markdown("---")
    st.subheader("🛠️ Engenharia de Atributos")
    usar_dif_temp = st.checkbox("Criar 'diferenca_temperatura'", value=True)
    usar_potencia = st.checkbox("Criar 'potencia' (torque * velocidade_rotacional)", value=True)

    st.markdown("---")
    st.subheader("🌐 API de Avaliação (opcional)")
    url_api = st.text_input("URL da API", value="", help="Endpoint para avaliar as previsões geradas no conjunto de teste. Deixe vazio para não submeter.")

# ---------------------------
# Carregamento dos dados
# ---------------------------
dados_treino, dados_teste, mensagens = carregar_dados(
    caminho_treino="bootcamp_train.csv",
    caminho_teste="bootcamp_test.csv",
    arquivo_treino=arquivo_treino,
    arquivo_teste=arquivo_teste
)

for tipo_msg, texto_msg in mensagens:
    # tipo_msg: "success", "info", "warning", "error"
    if hasattr(st, tipo_msg):
        getattr(st, tipo_msg)(texto_msg)
    else:
        st.info(texto_msg)

if dados_treino is None:
    st.error("Treino indisponível. Carregue o bootcamp_train.csv para continuar.")
    st.stop()

# ---------------------------
# Preparação / Feature Engineering
# ---------------------------
dados_treino_prep, codificador_tipo, codificador_falha = preparar_dados(
    dados_treino, treino=True, adicionar_dif_temp=usar_dif_temp, adicionar_potencia=usar_potencia
)

if dados_teste is not None:
    dados_teste_prep, _, _ = preparar_dados(
        dados_teste, treino=False, adicionar_dif_temp=usar_dif_temp, adicionar_potencia=usar_potencia,
        codificador_tipo=codificador_tipo, codificador_falha=codificador_falha
    )
else:
    dados_teste_prep = None

# ---------------------------
# Visão geral e EDA
# ---------------------------
st.header("📋 Visão Geral dos Dados")
with st.expander("Estrutura do conjunto de treino"):
    st.dataframe(dados_treino.head())
    st.write("Formato:", dados_treino.shape)

abas_eda = st.tabs(["Distribuições", "Correlação", "Falhas", "Tipos de Máquina"])

with abas_eda[0]:
    colunas_numericas = [c for c in [
        'temperatura_ar','temperatura_processo','umidade_relativa',
        'velocidade_rotacional','torque','desgaste_da_ferramenta',
        'diferenca_temperatura','potencia'
    ] if c in dados_treino_prep.columns]
    cols = st.columns(2)
    for i, col in enumerate(colunas_numericas):
        fig = plotar_histograma(dados_treino_prep, col, f"Distribuição de {col.replace('_',' ').title()}")
        cols[i % 2].plotly_chart(fig, use_container_width=True)

with abas_eda[1]:
    fig_corr = plotar_correlacao(dados_treino_prep)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Nenhuma coluna numérica disponível para correlação.")

with abas_eda[2]:
    if 'qualquer_falha' in dados_treino_prep.columns:
        proporcao = dados_treino_prep['qualquer_falha'].value_counts(normalize=True).sort_index()
        fig = px.bar(x=['Sem Falha','Com Falha'], y=(proporcao.values*100),
                     labels={'x':'Status','y':'Percentual (%)'}, title='Proporção de Falhas (binário)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Coluna 'qualquer_falha' não encontrada no treinamento.")

with abas_eda[3]:
    if 'tipo' in dados_treino_prep.columns:
        contagem = dados_treino_prep['tipo'].value_counts()
        st.plotly_chart(px.pie(values=contagem.values, names=contagem.index, hole=0.35, title='Distribuição por Tipo de Máquina'), use_container_width=True)
    else:
        st.info("Coluna 'tipo' não encontrada.")

# ---------------------------
# Treinamento
# ---------------------------
st.header("🏗️ Treinamento e Avaliação")

if tipo_modelagem.startswith("Binária"):
    alvo = 'qualquer_falha'
else:
    alvo = 'tipo_falha_codificado'
    if 'tipo_falha_codificado' not in dados_treino_prep.columns:
        st.warning("Rótulos específicos de falha não encontrados no treino. Voltando para binário.")
        alvo = 'qualquer_falha'

# Seleção de variáveis: remove colunas observacionais/labels
excluir = {'id','id_produto','tipo','qualquer_falha','tipo_falha_nome','tipo_falha_codificado',
           'FDF','FDC','FP','FTE','FA','falha_maquina'}
variaveis = [c for c in dados_treino_prep.columns if c not in excluir]

# seleciona apenas variáveis numéricas
X = dados_treino_prep[variaveis].select_dtypes(include=[np.number]).fillna(0)
y = dados_treino_prep[alvo].astype(int)

# Verifica número de classes para stratify
strat = y if y.nunique() > 1 else None

X_treino, X_valid, y_treino, y_valid = train_test_split(
    X, y, test_size=percentual_teste/100, random_state=int(semente_aleatoria), stratify=strat
)

modelo = RandomForestClassifier(n_estimators=400, random_state=int(semente_aleatoria), n_jobs=-1)
modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_valid)
acuracia = accuracy_score(y_valid, previsoes)
st.metric("Acurácia (validação)", f"{acuracia:.4f}")

with st.expander("Relatório de Classificação"):
    st.text(classification_report(y_valid, previsoes, zero_division=0))

st.plotly_chart(plotar_matriz_confusao(y_valid, previsoes), use_container_width=True)
st.plotly_chart(plotar_importancia_variaveis(modelo, variaveis), use_container_width=True)

# ---------------------------
# Prever / Submeter testes
# ---------------------------
if dados_teste_prep is not None:
    st.header("📤 Predições no conjunto de teste (bootcamp_test.csv)")
    
    # Alinha colunas: só pega as que existem em ambos
    colunas_comuns = [c for c in variaveis if c in dados_teste_prep.columns]
    X_test = dados_teste_prep[colunas_comuns].select_dtypes(include=[np.number]).fillna(0)
    
    preds_test = modelo.predict(X_test)
    proba_test = None
    if hasattr(modelo, "predict_proba") and modelo.n_classes_ <= 2:
        proba_test = modelo.predict_proba(X_test)[:, 1]
    elif hasattr(modelo, "predict_proba"):
        proba_test = np.max(modelo.predict_proba(X_test), axis=1)

    df_predicoes = pd.DataFrame({
        'id': dados_teste_prep.get('id', np.arange(len(preds_test))),
        'pred': preds_test
    })
    if proba_test is not None:
        df_predicoes['proba'] = proba_test


    st.dataframe(df_predicoes.head())

    if url_api:
        if st.button("📡 Enviar predições para API"):
            payload = montar_payload_api(dados_teste_prep, df_predicoes.rename(columns={'pred':'pred_qualquer_falha','proba':'proba_falha'}), tipo_modelagem)
            try:
                resp = requests.post(url_api, json=payload, timeout=30)
                st.write("Status:", resp.status_code)
                try:
                    st.json(resp.json())
                except Exception:
                    st.write(resp.text)
            except Exception as e:
                st.error(f"Erro ao enviar para a API: {e}")
else:
    st.info("Nenhum arquivo de teste fornecido (bootcamp_test.csv).")

st.markdown("---")
st.caption("Feito para o projeto do Bootcamp Ciência de Dados e IA, SESI/SENAI SC")



