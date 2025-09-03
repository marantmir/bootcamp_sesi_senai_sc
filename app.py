import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import requests

from utils import (
    carregar_dados,
    preparar_dados,
    plotar_histograma,
    plotar_correlacao,
    plotar_matriz_confusao,
    plotar_importancia_variaveis,
    montar_payload_api,
)

# -----------------------------------------------------
# Configuração da página
# -----------------------------------------------------
st.set_page_config(
    page_title="Sistema de Manutenção Preditiva",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔧 Sistema Inteligente de Manutenção Preditiva para Indústria 4.0")
st.markdown(
    """
    Este painel permite explorar dados de telemetria, treinar um modelo de classificação de falhas
    e validar o resultado em uma API externa.
    """
)
st.markdown("---")

# -----------------------------------------------------
# Barra lateral — Configurações
# -----------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configurações")

    st.subheader("📁 Fonte dos Dados")
    arquivo_treino = st.file_uploader("Selecione o Bootcamp_train.csv", type=["csv"], key="up_train")
    arquivo_teste = st.file_uploader("Selecione o Bootcamp_test.csv (opcional)", type=["csv"], key="up_test")

    st.markdown("---")

    st.subheader("🤖 Modelagem")
    tipo_modelagem = st.selectbox(
        "Tipo de Modelagem:",
        ["Binária (Qualquer Falha)", "Multiclasse (Tipos de Falha)"]
    )
    percentual_teste = st.slider("Percentual de Teste", 10, 40, 20, step=5)
    semente_aleatoria = st.slider("Semente Aleatória", 0, 999, 42)

    st.markdown("---")

    st.subheader("🛠️ Engenharia de Atributos")
    usar_dif_temp = st.checkbox("Criar 'diferença de temperatura'", value=True)
    usar_potencia = st.checkbox("Criar 'potência' (torque * velocidade rotacional)", value=True)

    st.markdown("---")

    st.subheader("🌐 API de Avaliação (opcional)")
    url_api = st.text_input(
        "URL da API",
        value="https://api-bootcamp-cdia.herokuapp.com/evaluate",
        help="Endpoint para avaliar as previsões geradas no conjunto de teste."
    )

# -----------------------------------------------------
# Carregamento dos dados
# -----------------------------------------------------
dados_treino, dados_teste, mensagens = carregar_dados(
    caminho_treino="bootcamp_train.csv",
    caminho_teste="bootcamp_test.csv",
    arquivo_treino=arquivo_treino,
    arquivo_teste=arquivo_teste,
)

for tipo_msg, texto_msg in mensagens:
    getattr(st, tipo_msg)(texto_msg)

if dados_treino is None:
    st.error("Treino indisponível. Carregue o Bootcamp_train.csv para continuar.")
    st.stop()

# -----------------------------------------------------
# Preparação / Feature Engineering
# -----------------------------------------------------
dados_treino_prep, codificador_tipo, codificador_falha = preparar_dados(
    dados_treino,
    treino=True,
    adicionar_dif_temp=usar_dif_temp,
    adicionar_potencia=usar_potencia,
)

# Se dados de teste existirem
if dados_teste is not None:
    dados_teste_prep, _, _ = preparar_dados(
        dados_teste,
        treino=False,
        adicionar_dif_temp=usar_dif_temp,
        adicionar_potencia=usar_potencia,
        codificador_tipo=codificador_tipo,
        codificador_falha=codificador_falha,
    )
else:
    dados_teste_prep = None

# -----------------------------------------------------
# Visão geral e EDA
# -----------------------------------------------------
st.header("📋 Visão Geral dos Dados")
with st.expander("Estrutura do conjunto de treino"):
    st.dataframe(dados_treino.head())
    st.write("Formato:", dados_treino.shape)

abas_eda = st.tabs(["Distribuições", "Correlação", "Falhas", "Tipos de Máquina"]) 

with abas_eda[0]:
    colunas_numericas = [
        c for c in [
            'temperatura_ar','temperatura_processo','umidade_relativa',
            'velocidade_rotacional','torque','desgaste_da_ferramenta',
            'diferenca_temperatura','potencia'
        ] if c in dados_treino_prep.columns
    ]
    cols = st.columns(2)
    for i, col in enumerate(colunas_numericas):
        fig = plotar_histograma(dados_treino_prep, col, f"Distribuição de {col.replace('_',' ').title()}")
        cols[i % 2].plotly_chart(fig, use_container_width=True)

with abas_eda[1]:
    fig_corr = plotar_correlacao(dados_treino_prep)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)

with abas_eda[2]:
    if 'qualquer_falha' in dados_treino_prep.columns:
        proporcao = dados_treino_prep['qualquer_falha'].value_counts(normalize=True).sort_index()
        fig = px.bar(
            x=['Sem Falha','Com Falha'],
            y=(proporcao.values*100),
            labels={'x':'Status','y':'Percentual (%)'},
            title='Proporção de Falhas (binário)'
        )
        st.plotly_chart(fig, use_container_width=True)

with abas_eda[3]:
    if 'tipo' in dados_treino_prep.columns:
        contagem = dados_treino_prep['tipo'].value_counts()
        st.plotly_chart(px.pie(values=contagem.values, names=contagem.index, hole=0.35,
                               title='Distribuição por Tipo de Máquina'), use_container_width=True)

# -----------------------------------------------------
# Treinamento
# -----------------------------------------------------
st.header("🏗️ Treinamento e Avaliação")

if tipo_modelagem.startswith("Binária"):
    alvo = 'qualquer_falha'
else:
    alvo = 'tipo_falha_codificado'
    if 'tipo_falha_codificado' not in dados_treino_prep.columns:
        st.warning("Rótulos específicos de falha não encontrados. Voltando para binário.")
        alvo = 'qualquer_falha'

variaveis = [
    c for c in dados_treino_prep.columns
    if c not in ['id','id_produto','tipo','qualquer_falha','tipo_falha_nome','tipo_falha_codificado',
                 'FDF','FDC','FP','FTE','FA']
]

X = dados_treino_prep[variaveis]
y = dados_treino_prep[alvo]

X_treino, X_valid, y_treino, y_valid = train_test_split(
    X, y, test_size=percentual_teste/100, random_state=semente_aleatoria,
    stratify=y if y.nunique()>1 else None
)

modelo = RandomForestClassifier(n_estimators=400, random_state=semente_aleatoria, n_jobs=-1)
modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_valid)
acuracia = accuracy_score(y_valid, previsoes)
st.metric("Acurácia (validação)", f"{acuracia:.4f}")

with st.expander("Relatório de Classificação"):
    st.text(classification_report(y_valid, previsoes))

st.plotly_chart(plotar_matriz_confusao(y_valid, previsoes), use_container_width=True)
st.plotly_chart(plotar_importancia_variaveis(modelo, variaveis), use_container_width=True)

## utils.py

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

# ------------------------------
# Carregamento de dados
# ------------------------------
def carregar_dados(caminho_treino, caminho_teste, arquivo_treino=None, arquivo_teste=None):
    mensagens = []

    dados_treino = None
    if arquivo_treino is not None:
        dados_treino = pd.read_csv(arquivo_treino)
        mensagens.append(("success", "Treino carregado via upload."))
    elif os.path.exists(caminho_treino):
        dados_treino = pd.read_csv(caminho_treino)
        mensagens.append(("success", "Treino carregado do diretório local."))

    dados_teste = None
    if arquivo_teste is not None:
        dados_teste = pd.read_csv(arquivo_teste)
        mensagens.append(("info", "Teste carregado via upload."))
    elif os.path.exists(caminho_teste):
        dados_teste = pd.read_csv(caminho_teste)
        mensagens.append(("info", "Teste carregado do diretório local."))

    return dados_treino, dados_teste, mensagens

# ------------------------------
# Preparação de dados
# ------------------------------
def preparar_dados(df, treino=True, adicionar_dif_temp=True, adicionar_potencia=True,
                   codificador_tipo=None, codificador_falha=None):
    dados = df.copy()

    if 'tipo' in dados.columns:
        if treino:
            codificador_tipo = LabelEncoder()
            dados['tipo_codificado'] = codificador_tipo.fit_transform(dados['tipo'])
        else:
            if codificador_tipo:
                dados['tipo_codificado'] = [
                    codificador_tipo.transform([x])[0] if x in codificador_tipo.classes_ else -1
                    for x in dados['tipo']
                ]

    if adicionar_dif_temp and {'temperatura_processo','temperatura_ar'}.issubset(dados.columns):
        dados['diferenca_temperatura'] = dados['temperatura_processo'] - dados['temperatura_ar']

    if adicionar_potencia and {'torque','velocidade_rotacional'}.issubset(dados.columns):
        dados['potencia'] = dados['torque'] * dados['velocidade_rotacional']

    if treino:
        col_falhas = [c for c in ['FDF','FDC','FP','FTE','FA'] if c in dados.columns]
        if col_falhas:
            dados['qualquer_falha'] = (dados[col_falhas].sum(axis=1) > 0).astype(int)
            dados['tipo_falha_nome'] = 'NF'
            for i, row in dados.iterrows():
                falhas = [c for c in col_falhas if row[c] == 1]
                if falhas:
                    dados.at[i,'tipo_falha_nome'] = falhas[0]
            codificador_falha = LabelEncoder()
            dados['tipo_falha_codificado'] = codificador_falha.fit_transform(dados['tipo_falha_nome'])
        elif 'falha_maquina' in dados.columns:
            dados['qualquer_falha'] = dados['falha_maquina']

    return dados, codificador_tipo, codificador_falha

# ------------------------------
# Visualizações
# ------------------------------
def plotar_histograma(df, coluna, titulo):
    fig = px.histogram(df, x=coluna, nbins=50, marginal="box", title=titulo)
    fig.update_layout(bargap=0.1)
    return fig


def plotar_correlacao(df):
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return None
    corr = num.corr()
    return px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)


def plotar_matriz_confusao(y_real, y_previsto):
    cm = confusion_matrix(y_real, y_previsto)
    etiquetas = [str(x) for x in sorted(set(list(y_real) + list(y_previsto)))]
    fig = px.imshow(cm, text_auto=True, aspect='auto', x=etiquetas, y=etiquetas,
                    title='Matriz de Confusão', color_continuous_scale='Blues')
    fig.update_xaxes(title="Previsto")
    fig.update_yaxes(title="Real")
    return fig


def plotar_importancia_variaveis(modelo, variaveis):
    if hasattr(modelo,'feature_importances_'):
        imp = pd.DataFrame({'variavel':variaveis,'importancia':modelo.feature_importances_})
        imp = imp.sort_values('importancia')
        return px.bar(imp, x='importancia', y='variavel', orientation='h', title='Importância das Variáveis')
    return px.bar(title='Importância não disponível')

# ------------------------------
# Submissão API
# ------------------------------
def montar_payload_api(df_teste, df_predicoes, tipo_modelagem):
    payload = {'predictions': []}
    if 'id' in df_teste.columns:
        base = df_teste[['id']].merge(df_predicoes, on='id', how='left')
    else:
        base = df_predicoes.copy()
        base.insert(0,'id', np.arange(len(base)))

    if tipo_modelagem.startswith('Binária'):
        for _, r in base.iterrows():
            payload['predictions'].append({
                'id': int(r['id']),
                'label': int(r.get('pred_qualquer_falha',0)),
                'prob': float(r.get('proba_falha',0.0))
            })
    else:
        for _, r in base.iterrows():
            payload['predictions'].append({
                'id': int(r['id']),
                'label': int(r.get('pred_tipo_falha_cod',0))
            })
    return payload

