import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import plotly.express as px
import os

# Ordem de colunas exigida pela API
FALHAS = ['FDF', 'FDC', 'FP', 'FTE', 'FA']


def carregar_dados(caminho_treino, caminho_teste, arquivo_treino=None, arquivo_teste=None):
    mensagens, dados_treino, dados_teste = [], None, None
    if arquivo_treino is not None:
        dados_treino = pd.read_csv(arquivo_treino)
        mensagens.append(("success", "Treino carregado via upload."))
    elif os.path.exists(caminho_treino):
        dados_treino = pd.read_csv(caminho_treino)
        mensagens.append(("success", "Treino carregado do diretório local."))
    if arquivo_teste is not None:
        dados_teste = pd.read_csv(arquivo_teste)
        mensagens.append(("info", "Teste carregado via upload."))
    elif os.path.exists(caminho_teste):
        dados_teste = pd.read_csv(caminho_teste)
        mensagens.append(("info", "Teste carregado do diretório local."))
    return dados_treino, dados_teste, mensagens


def preparar_dados(df, treino=True, adicionar_dif_temp=True, adicionar_potencia=True,
                   codificador_tipo=None, codificador_falha=None):
    dados = df.copy()

    if 'tipo' in dados.columns:
        if treino:
            codificador_tipo = LabelEncoder()
            dados['tipo_codificado'] = codificador_tipo.fit_transform(dados['tipo'])
        elif codificador_tipo:
            dados['tipo_codificado'] = [
                codificador_tipo.transform([x])[0] if x in codificador_tipo.classes_ else -1
                for x in dados['tipo']
            ]

    if adicionar_dif_temp and {'temperatura_processo', 'temperatura_ar'}.issubset(dados.columns):
        dados['diferenca_temperatura'] = dados['temperatura_processo'] - dados['temperatura_ar']
    if adicionar_potencia and {'torque', 'velocidade_rotacional'}.issubset(dados.columns):
        dados['potencia'] = dados['torque'] * dados['velocidade_rotacional']

    if treino:
        col_falhas = [c for c in FALHAS if c in dados.columns]
        if col_falhas:
            dados['qualquer_falha'] = (dados[col_falhas].sum(axis=1) > 0).astype(int)
            dados['tipo_falha_nome'] = 'NF'
            for i, row in dados.iterrows():
                pos = [c for c in col_falhas if row.get(c, 0) == 1]
                if pos:
                    dados.at[i, 'tipo_falha_nome'] = pos[0]
            codificador_falha = LabelEncoder()
            dados['tipo_falha_codificado'] = codificador_falha.fit_transform(dados['tipo_falha_nome'])
        elif 'falha_maquina' in dados.columns:
            dados['qualquer_falha'] = (
                pd.to_numeric(dados['falha_maquina'], errors='coerce').fillna(0).astype(int)
            )

    return dados, codificador_tipo, codificador_falha


def garantir_numerico(df):
    return df.select_dtypes(include=[np.number]).fillna(0)


def one_hot_falhas_from_multiclasse(pred_cod, codificador_falha):
    nomes = codificador_falha.inverse_transform(pred_cod)
    out = pd.DataFrame(0, index=np.arange(len(nomes)), columns=FALHAS)
    for i, nome in enumerate(nomes):
        if nome in FALHAS:
            out.at[i, nome] = 1
    return out


def one_hot_falhas_from_binario(pred_bin):
    out = pd.DataFrame(0, index=np.arange(len(pred_bin)), columns=FALHAS)
    idx = np.where(np.array(pred_bin) == 1)[0]
    if len(idx):
        out.iloc[idx, out.columns.get_loc('FDF')] = 1
    return out


def submission_csv(df_multilabel):
    df = pd.DataFrame(0, index=df_multilabel.index, columns=FALHAS)
    for c in FALHAS:
        if c in df_multilabel.columns:
            df[c] = (df_multilabel[c].astype(float) > 0.5).astype(int)
    return df


def plotar_matriz_confusao(y_real, y_previsto):
    cm = confusion_matrix(y_real, y_previsto)
    etiquetas = [str(x) for x in sorted(set(list(y_real) + list(y_previsto)))]
    fig = px.imshow(cm, text_auto=True, aspect='auto', x=etiquetas, y=etiquetas,
                    title='Matriz de Confusão', color_continuous_scale='Blues')
    fig.update_xaxes(title="Previsto")
    fig.update_yaxes(title="Real")
    return fig


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


def plotar_importancia_variaveis(modelo, variaveis):
    if hasattr(modelo, 'feature_importances_'):
        imp = pd.DataFrame({'variavel': variaveis, 'importancia': modelo.feature_importances_})
        imp = imp.sort_values('importancia')
        return px.bar(imp, x='importancia', y='variavel', orientation='h', title='Importância das Variáveis')
    return px.bar(title='Importância não disponível')
