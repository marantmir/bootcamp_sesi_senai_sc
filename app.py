import os, io, numpy as np, pandas as pd, streamlit as st, requests
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
    garantir_numerico,
    one_hot_falhas_from_multiclasse,
    one_hot_falhas_from_binario,
    submission_csv,
    FALHAS
)

# ---------------- Página / Sidebar ----------------
st.set_page_config(page_title="Sistema de Manutenção Preditiva", page_icon="🔧", layout="wide")
st.title("🔧 Sistema Inteligente de Manutenção Preditiva")

with st.sidebar:
    st.header("⚙️ Configurações")
    arquivo_treino = st.file_uploader("Selecione o Bootcamp_train.csv", type=["csv"])
    arquivo_teste = st.file_uploader("Selecione o Bootcamp_test.csv (opcional)", type=["csv"])
    tipo_modelagem = st.selectbox("Tipo de Modelagem:", ["Binária (Qualquer Falha)", "Multiclasse (Tipos de Falha)"])
    percentual_teste = st.slider("Percentual de Teste", 10, 40, 20, step=5)
    semente_aleatoria = st.slider("Semente Aleatória", 0, 999, 42)
    usar_dif_temp = st.checkbox("Criar 'diferença de temperatura'", value=True)
    usar_potencia = st.checkbox("Criar 'potência' (torque * velocidade rotacional)", value=True)
    threshold_api = st.slider("Threshold (API)", 0.0, 1.0, 0.5, 0.05)
    TOKEN_API = "b611eddf6f51841fb1849dde92b2013f5bc33ca3e4a5ceb645326c22a8e3e4f7"
    URL_API = "http://34.193.187.218:5000/evaluate/multilabel_metrics"

# ---------------- Carregamento ----------------
dados_treino, dados_teste, mensagens = carregar_dados(
    caminho_treino="bootcamp_train.csv",
    caminho_teste="bootcamp_test.csv",
    arquivo_treino=arquivo_treino,
    arquivo_teste=arquivo_teste,
)
for tipo_msg, texto_msg in mensagens:
    getattr(st, tipo_msg)(texto_msg)
if dados_treino is None:
    st.error("Treino indisponível. Carregue o Bootcamp_train.csv.")
    st.stop()

# ---------------- Preparação ----------------
dados_treino_prep, codificador_tipo, codificador_falha = preparar_dados(
    dados_treino, treino=True, adicionar_dif_temp=usar_dif_temp, adicionar_potencia=usar_potencia
)
dados_teste_prep = None
if dados_teste is not None:
    dados_teste_prep, _, _ = preparar_dados(
        dados_teste, treino=False, adicionar_dif_temp=usar_dif_temp, adicionar_potencia=usar_potencia,
        codificador_tipo=codificador_tipo, codificador_falha=codificador_falha
    )

# ---------------- Treinamento ----------------
alvo = 'qualquer_falha' if tipo_modelagem.startswith("Binária") else 'tipo_falha_codificado'
if alvo == 'tipo_falha_codificado' and 'tipo_falha_codificado' not in dados_treino_prep.columns:
    st.warning("Rótulos de falha não encontrados. Voltando para binário.")
    alvo = 'qualquer_falha'

excluir = {'id', 'id_produto', 'tipo', 'qualquer_falha', 'tipo_falha_nome',
           'tipo_falha_codificado', 'FDF', 'FDC', 'FP', 'FTE', 'FA', 'falha_maquina'}
variaveis = [c for c in dados_treino_prep.columns if c not in excluir]
X = garantir_numerico(dados_treino_prep[variaveis])
y = dados_treino_prep[alvo].astype(int)

X_treino, X_valid, y_treino, y_valid = train_test_split(
    X, y, test_size=percentual_teste / 100, random_state=semente_aleatoria,
    stratify=y if y.nunique() > 1 else None
)

modelo = RandomForestClassifier(n_estimators=400, random_state=semente_aleatoria, n_jobs=-1)
modelo.fit(X_treino, y_treino)
pred_valid = modelo.predict(X_valid)
st.metric("Acurácia (validação)", f"{accuracy_score(y_valid, pred_valid):.4f}")
with st.expander("Relatório de Classificação"):
    st.text(classification_report(y_valid, pred_valid))
st.plotly_chart(plotar_matriz_confusao(y_valid, pred_valid), use_container_width=True)
st.plotly_chart(plotar_importancia_variaveis(modelo, X.columns.tolist()), use_container_width=True)

# ---------------- Predição no Teste + Envio API ----------------
if dados_teste_prep is not None:
    st.header("📤 Predições no conjunto de teste")

    colunas_comuns = [c for c in variaveis if c in dados_teste_prep.columns]
    X_test = garantir_numerico(dados_teste_prep[colunas_comuns])
    preds_test = modelo.predict(X_test)

    if alvo == 'tipo_falha_codificado':
        df_multi = one_hot_falhas_from_multiclasse(preds_test, codificador_falha)
    else:
        df_multi = one_hot_falhas_from_binario(preds_test)

    df_submit = submission_csv(df_multi)

    # Validação rigorosa do formato antes do envio
    colunas_esperadas = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
    if not all(col in df_submit.columns for col in colunas_esperadas):
        st.error("CSV deve conter exatamente as colunas: FDF, FDC, FP, FTE, FA")
    else:
        # Garantir que apenas as colunas necessárias estejam presentes
        df_submit = df_submit[colunas_esperadas].copy()
        
        # Forçar conversão para inteiros 0 ou 1
        for col in colunas_esperadas:
            df_submit[col] = df_submit[col].astype(int)
            # Garantir que são apenas 0 ou 1
            df_submit[col] = df_submit[col].clip(0, 1)
        
        # Verificar se o número de linhas corresponde ao arquivo de teste original
        if dados_teste is not None and len(df_submit) != len(dados_teste):
            st.error(f"Número de linhas incorreto. Esperado: {len(dados_teste)}, Atual: {len(df_submit)}")
        else:
            st.subheader("Prévia do CSV para API")
            st.dataframe(df_submit.head())
            
            # Mostrar estatísticas de validação
            st.write("**Validação do formato:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Linhas", len(df_submit))
            with col2:
                st.metric("Colunas", len(df_submit.columns))
            with col3:
                valores_unicos = set()
                for col in colunas_esperadas:
                    valores_unicos.update(df_submit[col].unique())
                st.metric("Valores únicos", f"{sorted(valores_unicos)}")

            if st.button("📡 Enviar predições para API"):
                try:
                    # Criar CSV com formatação específica
                    csv_buffer = io.StringIO()
                    df_submit.to_csv(csv_buffer, index=False, lineterminator='\n')
                    csv_content = csv_buffer.getvalue()
                    
                    # Log do conteúdo para debug
                    with st.expander("Debug - Primeiras linhas do CSV"):
                        st.text(csv_content[:500] + "..." if len(csv_content) > 500 else csv_content)

                    headers = {"X-API-Key": TOKEN_API}
                    params = {"threshold": float(threshold_api)}
                    files = {"file": ("submission.csv", csv_content, "text/csv")}

                    with st.spinner("Enviando para API..."):
                        resp = requests.post(URL_API, headers=headers, files=files, params=params, timeout=120)
                    
                    st.write("**Status da resposta:**", resp.status_code)
                    
                    if resp.status_code == 200:
                        st.success("✅ Predições enviadas com sucesso!")
                        if resp.headers.get("content-type", "").startswith("application/json"):
                            resultado = resp.json()
                            st.json(resultado)
                            
                            # Mostrar métricas de forma mais organizada se disponível
                            if isinstance(resultado, dict):
                                st.subheader("📊 Métricas de Performance")
                                for metrica, valor in resultado.items():
                                    if isinstance(valor, (int, float)):
                                        st.metric(metrica.replace('_', ' ').title(), f"{valor:.4f}")
                        else:
                            st.write("Resposta da API:", resp.text)
                    else:
                        st.error(f"❌ Erro na API (Status {resp.status_code})")
                        try:
                            if resp.headers.get("content-type", "").startswith("application/json"):
                                erro_detalhes = resp.json()
                                st.json(erro_detalhes)
                            else:
                                st.write("Detalhes do erro:", resp.text)
                        except:
                            st.write("Não foi possível decodificar a resposta de erro")
                        
                        st.info("💡 Dicas para resolver:")
                        st.write("- Verifique se o arquivo de teste tem o mesmo número de linhas")
                        st.write("- Confirme que todas as predições são 0 ou 1")
                        st.write("- Verifique se há valores NaN ou missing")

                except requests.exceptions.Timeout:
                    st.error("⏰ Timeout na conexão com a API. Tente novamente.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Erro de conexão com a API. Verifique sua internet.")
                except Exception as e:
                    st.error(f"❌ Erro inesperado: {str(e)}")
                    st.write("Tipo do erro:", type(e).__name__)
