# utils.py
import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# ------------------------------
# Carregamento de dados
# ------------------------------
def carregar_dados(caminho_treino="bootcamp_train.csv", caminho_teste="bootcamp_test.csv",
                   arquivo_treino=None, arquivo_teste=None):
    mensagens = []
    dados_treino = None
    dados_teste = None

    # Treino: prioriza upload do usuário
    if arquivo_treino is not None:
        try:
            dados_treino = pd.read_csv(arquivo_treino)
            mensagens.append(("success", "Treino carregado via upload."))
        except Exception as e:
            mensagens.append(("error", f"Falha ao ler arquivo de treino: {e}"))
    elif os.path.exists(caminho_treino):
        try:
            dados_treino = pd.read_csv(caminho_treino)
            mensagens.append(("success", "Treino carregado do diretório local."))
        except Exception as e:
            mensagens.append(("error", f"Falha ao ler arquivo local de treino: {e}"))

    # Teste (opcional)
    if arquivo_teste is not None:
        try:
            dados_teste = pd.read_csv(arquivo_teste)
            mensagens.append(("info", "Teste carregado via upload."))
        except Exception as e:
            mensagens.append(("warning", f"Falha ao ler arquivo de teste: {e}"))
    elif os.path.exists(caminho_teste):
        try:
            dados_teste = pd.read_csv(caminho_teste)
            mensagens.append(("info", "Teste carregado do diretório local."))
        except Exception as e:
            mensagens.append(("warning", f"Falha ao ler arquivo local de teste: {e}"))

    return dados_treino, dados_teste, mensagens

# ------------------------------
# Preparação de dados
# ------------------------------
def preparar_dados(df, treino=True, adicionar_dif_temp=True, adicionar_potencia=True,
                   codificador_tipo=None, codificador_falha=None):
    """
    Retorna: (dados_transformados, codificador_tipo, codificador_falha)
    - Se treino=True, treina os LabelEncoders e gera colunas 'tipo_codificado' e 'tipo_falha_codificado'
    - Se treino=False, utiliza os codificadores passados (se houver) e marca classes desconhecidas como -1
    """
    dados = df.copy()
    if dados is None:
        return None, codificador_tipo, codificador_falha

    # codificação do tipo
    if 'tipo' in dados.columns:
        if treino:
            codificador_tipo = LabelEncoder()
            dados['tipo_codificado'] = codificador_tipo.fit_transform(dados['tipo'].astype(str))
        else:
            if codificador_tipo is not None:
                # transforma com cautela (valores desconhecidos -> -1)
                classes = list(codificador_tipo.classes_)
                def safe_transform(x):
                    try:
                        return int(codificador_tipo.transform([x])[0])
                    except Exception:
                        return -1
                dados['tipo_codificado'] = dados['tipo'].apply(safe_transform)
            else:
                dados['tipo_codificado'] = -1

    # features derivadas
    if adicionar_dif_temp and {'temperatura_processo','temperatura_ar'}.issubset(dados.columns):
        dados['diferenca_temperatura'] = dados['temperatura_processo'] - dados['temperatura_ar']

    if adicionar_potencia and {'torque','velocidade_rotacional'}.issubset(dados.columns):
        dados['potencia'] = dados['torque'] * dados['velocidade_rotacional']

    # tratamento das falhas (apenas no treino tem labels)
    if treino:
        col_falhas = [c for c in ['FDF','FDC','FP','FTE','FA'] if c in dados.columns]
        if col_falhas:
            dados['qualquer_falha'] = (dados[col_falhas].sum(axis=1) > 0).astype(int)
            # determina nome da primeira falha presente (se houver)
            dados['tipo_falha_nome'] = 'NF'
            for i, row in dados.iterrows():
                falhas = [c for c in col_falhas if row.get(c, 0) == 1]
                if falhas:
                    dados.at[i, 'tipo_falha_nome'] = falhas[0]
            codificador_falha = LabelEncoder()
            dados['tipo_falha_codificado'] = codificador_falha.fit_transform(dados['tipo_falha_nome'])
        elif 'falha_maquina' in dados.columns:
        # converte valores para número (ex: 0/1), trata inválidos como 0
            dados['qualquer_falha'] = (
                pd.to_numeric(dados['falha_maquina'], errors='coerce')
                .fillna(0)
                .astype(int)
            )


    return dados, codificador_tipo, codificador_falha

# ------------------------------
# Visualizações
# ------------------------------
def plotar_histograma(df, coluna, titulo):
    if coluna not in df.columns:
        return px.histogram(title=f"{titulo} (coluna não encontrada)")
    fig = px.histogram(df, x=coluna, nbins=50, marginal="box", title=titulo)
    fig.update_layout(bargap=0.1)
    return fig

def plotar_correlacao(df):
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return None
    corr = num.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", zmin=-1, zmax=1)
    fig.update_layout(title="Matriz de Correlação")
    return fig

def plotar_matriz_confusao(y_real, y_previsto):
    try:
        cm = confusion_matrix(y_real, y_previsto)
        etiquetas = [str(x) for x in sorted(set(list(y_real) + list(y_previsto)))]
        fig = px.imshow(cm, text_auto=True, aspect='auto', x=etiquetas, y=etiquetas,
                        title='Matriz de Confusão')
        fig.update_xaxes(title="Previsto")
        fig.update_yaxes(title="Real")
        return fig
    except Exception:
        return px.imshow([[0]], title="Matriz de Confusão (erro)")

def plotar_importancia_variaveis(modelo, variaveis):
    try:
        if hasattr(modelo, 'feature_importances_'):
            imp = pd.DataFrame({'variavel': variaveis, 'importancia': modelo.feature_importances_})
            imp = imp.sort_values('importancia', ascending=True)
            fig = px.bar(imp, x='importancia', y='variavel', orientation='h', title='Importância das Variáveis')
            return fig
        if hasattr(modelo, 'coef_'):
            coef = np.ravel(modelo.coef_)
            imp = pd.DataFrame({'variavel': variaveis, 'importancia': np.abs(coef)})
            imp = imp.sort_values('importancia', ascending=True)
            return px.bar(imp, x='importancia', y='variavel', orientation='h', title='Importância (coef)')
    except Exception:
        pass
    return px.bar(title='Importância não disponível')

# ------------------------------
# Montagem payload API
# ------------------------------
def montar_payload_api(df_teste, df_predicoes, tipo_modelagem):
    payload = {'predictions': []}
    if df_predicoes is None:
        return payload

    base = df_predicoes.copy()
    # tenta alinhar com ids do df_teste se possível
    if df_teste is not None and 'id' in df_teste.columns and 'id' in base.columns:
        base = df_teste[['id']].merge(base, on='id', how='left')

    if 'id' not in base.columns:
        base.insert(0, 'id', np.arange(len(base)))

    if tipo_modelagem.startswith('Binária'):
        for _, r in base.iterrows():
            payload['predictions'].append({
                'id': int(r['id']),
                'label': int(r.get('pred_qualquer_falha', r.get('pred', 0))),
                'prob': float(r.get('proba_falha', r.get('proba', 0.0)))
            })
    else:
        for _, r in base.iterrows():
            payload['predictions'].append({
                'id': int(r['id']),
                'label': int(r.get('pred_tipo_falha_cod', r.get('pred', 0)))
            })
    return payload
