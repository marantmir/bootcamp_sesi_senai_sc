import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
import requests
import joblib
import os
import json
import time
import subprocess
import pkg_resources

# Verificar e instalar dependências se necessário
required = {'scikit-learn==1.3.2', 'numpy==1.24.3', 'cython==0.29.36'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

# Configuração da página
st.set_page_config(
    page_title="Sistema de Manutenção Preditiva",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🔧 Sistema Inteligente de Manutenção Preditiva")
st.markdown("---")

# Verificar se os arquivos existem localmente
arquivo_treino_existe = os.path.exists("bootcamp_train.csv")
arquivo_teste_existe = os.path.exists("bootcamp_test.csv")

# Barra lateral para configurações
with st.sidebar:
    st.header("Configurações do Projeto")
    
    # Informações sobre arquivos
    st.subheader("Status dos Arquivos")
    if arquivo_treino_existe:
        st.success("✅ bootcamp_train.csv encontrado")
    else:
        st.error("❌ bootcamp_train.csv não encontrado")
        
    if arquivo_teste_existe:
        st.success("✅ bootcamp_test.csv encontrado")
    else:
        st.warning("⚠️ bootcamp_test.csv não encontrado")
    
    # Configurações de análise
    st.subheader("Configurações de Análise")
    opcao_analise = st.selectbox(
        "Tipo de Análise Exploratória:",
        ["Geral", "Por Tipo de Máquina", "Por Tipo de Falha"]
    )
    
    # Configurações do modelo
    st.subheader("Configurações do Modelo")
    tipo_modelagem = st.selectbox(
        "Tipo de Modelagem:",
        ["Binária (Falha vs Não Falha)", "Multiclasse (Tipo de Falha)"]
    )
    
    tamanho_teste = st.slider("Percentual para Teste:", 10, 40, 20)
    estado_aleatorio = st.slider("Semente Aleatória:", 0, 100, 42)
    
    # Opções de engenharia de características
    st.subheader("Engenharia de Características")
    usar_diferenca_temperatura = st.checkbox("Usar diferença de temperatura", value=True)
    usar_potencia = st.checkbox("Usar cálculo de potência", value=True)
    
    # Configuração da API
    st.subheader("Configurações da API")
    url_api = st.text_input("URL da API de avaliação:", "https://api-bootcamp-cdia.herokuapp.com/evaluate")

# Funções auxiliares
@st.cache_data
def carregar_dados():
    """Carrega os dados dos arquivos CSV locais"""
    try:
        dados_treino = pd.read_csv("bootcamp_train.csv")
        dados_teste = pd.read_csv("bootcamp_test.csv") if arquivo_teste_existe else None
        return dados_treino, dados_teste
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        return None, None

def preprocessar_dados(df, eh_treino=True, usar_diff_temp=True, usar_pot=True):
    """Pré-processa os dados"""
    df_processado = df.copy()
    
    # Codificar tipo de máquina
    codificador = LabelEncoder()
    df_processado['tipo_codificado'] = codificador.fit_transform(df_processado['tipo'])
    
    # Calcular diferença de temperatura (se habilitado)
    if usar_diff_temp:
        df_processado['diferenca_temperatura'] = df_processado['temperatura_processo'] - df_processado['temperatura_ar']
    
    # Calcular potência (se habilitado)
    if usar_pot:
        df_processado['potencia'] = df_processado['torque'] * df_processado['velocidade_rotacional']
    
    if eh_treino:
        # Calcular se há qualquer tipo de falha
        colunas_falhas = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
        df_processado['qualquer_falha'] = df_processado[colunas_falhas].max(axis=1)
    
    return df_processado

def plotar_distribuicoes(df, coluna, titulo):
    """Plota distribuições dos dados"""
    figura = px.histogram(df, x=coluna, title=titulo, nbins=50, marginal="box")
    figura.update_layout(bargap=0.1)
    return figura

def plotar_matriz_correlacao(df):
    """Plota matriz de correlação"""
    df_numerico = df.select_dtypes(include=[np.number])
    correlacao = df_numerico.corr()
    
    figura = go.Figure(data=go.Heatmap(
        z=correlacao.values,
        x=correlacao.columns,
        y=correlacao.index,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlação"),
        hoverongaps=False,
        hovertemplate='<b>X</b>: %{x}<br><b>Y</b>: %{y}<br><b>Correlação</b>: %{z:.2f}<extra></extra>'
    ))
    
    figura.update_layout(
        title="Matriz de Correlação",
        width=800,
        height=800
    )
    
    return figura

def plotar_matriz_confusao(y_real, y_previsto, rotulos):
    """Plota matriz de confusão"""
    matriz_confusao = confusion_matrix(y_real, y_previsto)
    figura = px.imshow(
        matriz_confusao, 
        text_auto=True,
        aspect="auto",
        x=rotulos,
        y=rotulos,
        title="Matriz de Confusão",
        color_continuous_scale='Blues'
    )
    figura.update_xaxes(title="Previsto")
    figura.update_yaxes(title="Real")
    return figura

def plotar_importancia_caracteristicas(importancia_caracteristicas):
    """Plota importância das características"""
    figura = px.bar(
        importancia_caracteristicas, 
        x='importancia', 
        y='caracteristica',
        title='Importância das Características no Modelo',
        orientation='h',
        color='importancia',
        color_continuous_scale='viridis'
    )
    figura.update_layout(yaxis={'categoryorder':'total ascending'})
    return figura

def enviar_para_api(dados_envio, url_api):
    """Envia dados para a API e retorna a resposta"""
    try:
        with st.spinner('Enviando previsões para a API...'):
            resposta = requests.post(url_api, json=dados_envio)
        
        if resposta.status_code == 200:
            return True, resposta.json()
        else:
            return False, f"Erro HTTP {resposta.status_code}: {resposta.text}"
    except Exception as e:
        return False, f"Erro de conexão: {str(e)}"

def exibir_resultados_api(resultados):
    """Exibe os resultados retornados pela API de forma organizada"""
    if not isinstance(resultados, dict):
        st.error("Formato de resposta da API inválido")
        return
    
    st.subheader("📊 Resultados da Validação pela API")
    
    # Métricas gerais
    if 'overall_accuracy' in resultados:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Acurácia Geral", f"{resultados['overall_accuracy']:.4f}")
        with col2:
            st.metric("Precision Média", f"{resultados.get('mean_precision', 0):.4f}")
        with col3:
            st.metric("Recall Médio", f"{resultados.get('mean_recall', 0):.4f}")
    
    # Matriz de confusão
    if 'confusion_matrix' in resultados:
        st.subheader("Matriz de Confusão (API)")
        fig = px.imshow(
            resultados['confusion_matrix'], 
            text_auto=True,
            aspect="auto",
            title="Matriz de Confusão - Validação API",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Métricas por classe
    if 'class_metrics' in resultados:
        st.subheader("Métricas por Classe")
        metricas_df = pd.DataFrame(resultados['class_metrics']).T
        st.dataframe(metricas_df)
    
    # Curva ROC (se disponível)
    if 'roc_auc' in resultados:
        st.subheader("Curva ROC")
        st.write(f"AUC Score: {resultados['roc_auc']:.4f}")

# Carregar dados
if arquivo_treino_existe:
    dados_treino, dados_teste = carregar_dados()
    
    if dados_treino is not None:
        dados_processados = preprocessar_dados(dados_treino, eh_treino=True, 
                                              usar_diff_temp=usar_diferenca_temperatura, 
                                              usar_pot=usar_potencia)
        
        # Exibir informações básicas dos dados
        st.header("📊 Análise Exploratória dos Dados")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Amostras", dados_treino.shape[0])
        with col2:
            st.metric("Total de Características", dados_treino.shape[1])
        with col3:
            falhas = dados_processados['qualquer_falha'].sum()
            st.metric("Máquinas com Falha", f"{falhas} ({falhas/dados_treino.shape[0]*100:.1f}%)")
        with col4:
            tipos_maquina = dados_treino['tipo'].nunique()
            st.metric("Tipos de Máquina", tipos_maquina)
        
        # Abas para diferentes análises
        aba1, aba2, aba3, aba4, aba5 = st.tabs(["Visão Geral", "Distribuições", "Correlações", "Análise de Falhas", "Tipos de Máquina"])
        
        with aba1:
            st.subheader("Visualização dos Dados")
            st.dataframe(dados_treino.head(10))
            
            st.subheader("Informações Estatísticas")
            st.dataframe(dados_treino.describe())
            
            st.subheader("Tipos de Dados e Valores Nulos")
            info_df = pd.DataFrame({
                'Tipo': dados_treino.dtypes,
                'Valores Nulos': dados_treino.isnull().sum(),
                'Percentual Nulos': (dados_treino.isnull().sum() / len(dados_treino)) * 100
            })
            st.dataframe(info_df)
        
        with aba2:
            st.subheader("Distribuição das Variáveis Numéricas")
            
            colunas_numericas = ['temperatura_ar', 'temperatura_processo', 'umidade_relativa', 
                               'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta']
            
            colunas_por_linha = 2
            for i in range(0, len(colunas_numericas), colunas_por_linha):
                colunas = st.columns(colunas_por_linha)
                for j, coluna in enumerate(colunas_numericas[i:i+colunas_por_linha]):
                    with colunas[j]:
                        figura = plotar_distribuicoes(dados_treino, coluna, f"Distribuição de {coluna}")
                        st.plotly_chart(figura, use_container_width=True)
        
        with aba3:
            st.subheader("Matriz de Correlação")
            figura = plotar_matriz_correlacao(dados_processados)
            st.plotly_chart(figura, use_container_width=True)
        
        with aba4:
            st.subheader("Análise de Falhas")
            
            # Contagem de falhas por tipo
            colunas_falhas = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
            contagem_falhas = dados_treino[colunas_falhas].sum()
            
            figura = px.pie(
                values=contagem_falhas.values,
                names=contagem_falhas.index,
                title="Distribuição de Tipos de Falha"
            )
            st.plotly_chart(figura, use_container_width=True)
            
            # Relação entre variáveis e falhas
            st.subheader("Relação entre Variáveis e Falhas")
            variavel_selecionada = st.selectbox("Selecione a variável:", colunas_numericas, key="var_select")
            
            figura = px.box(
                dados_treino, 
                x='falha_maquina', 
                y=variavel_selecionada,
                title=f"{variavel_selecionada} vs Falha de Máquina",
                color='falha_maquina'
            )
            st.plotly_chart(figura, use_container_width=True)
        
        with aba5:
            st.subheader("Análise por Tipo de Máquina")
            
            # Distribuição por tipo de máquina
            contagem_tipos = dados_treino['tipo'].value_counts()
            figura = px.pie(
                values=contagem_tipos.values,
                names=contagem_tipos.index,
                title="Distribuição por Tipo de Máquina"
            )
            st.plotly_chart(figura, use_container_width=True)
            
            # Falhas por tipo de máquina
            falhas_por_tipo = dados_treino.groupby('tipo')['falha_maquina'].mean() * 100
            figura = px.bar(
                x=falhas_por_tipo.index,
                y=falhas_por_tipo.values,
                title="Percentual de Falhas por Tipo de Máquina",
                labels={'x': 'Tipo de Máquina', 'y': 'Percentual de Falhas (%)'}
            )
            st.plotly_chart(figura, use_container_width=True)
        
        # Divisão dos dados e treinamento do modelo
        st.header("🤖 Modelagem Preditiva")
        
        # Selecionar características baseado nas opções
        caracteristicas = ['tipo_codificado', 'temperatura_ar', 'temperatura_processo', 'umidade_relativa',
                         'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta']
        
        if usar_diferenca_temperatura:
            caracteristicas.append('diferenca_temperatura')
        if usar_potencia:
            caracteristicas.append('potencia')
        
        if tipo_modelagem == "Binária (Falha vs Não Falha)":
            alvo = 'qualquer_falha'
            modelo = RandomForestClassifier(n_estimators=100, random_state=estado_aleatorio, class_weight='balanced')
            tipo_problema = 'binario'
        else:  # Multiclasse
            # Criar uma coluna com o tipo de falha predominante
            colunas_falhas = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
            dados_processados['tipo_falha'] = dados_processados[colunas_falhas].idxmax(axis=1)
            # Se não há falha, marca como 'NF'
            dados_processados.loc[dados_processados['qualquer_falha'] == 0, 'tipo_falha'] = 'NF'
            
            codificador_falha = LabelEncoder()
            dados_processados['tipo_falha_codificado'] = codificador_falha.fit_transform(dados_processados['tipo_falha'])
            
            alvo = 'tipo_falha_codificado'
            modelo = RandomForestClassifier(n_estimators=100, random_state=estado_aleatorio, class_weight='balanced')
            tipo_problema = 'multiclasse'
        
        # Dividir dados em treino e teste
        X = dados_processados[caracteristicas]
        y = dados_processados[alvo]
        
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y, test_size=tamanho_teste/100, random_state=estado_aleatorio, stratify=y
        )
        
        # Treinar modelo
        with st.spinner('Treinando modelo...'):
            modelo.fit(X_treino, y_treino)
        
        # Fazer previsões
        y_previsto = modelo.predict(X_teste)
        y_probabilidade = modelo.predict_proba(X_teste)
        
        # Avaliar modelo
        acuracia = accuracy_score(y_teste, y_previsto)
        
        st.subheader("Resultados do Modelo (Validação Interna)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Acurácia do Modelo", f"{acuracia:.4f}")
        
        with col2:
            st.metric("Amostras de Treino", X_treino.shape[0])
        
        with col3:
            st.metric("Amostras de Teste", X_teste.shape[0])
        
        # Matriz de confusão
        if tipo_problema == 'binario':
            rotulos = ['Sem Falha', 'Com Falha']
        else:
            rotulos = list(codificador_falha.classes_)
        
        figura = plotar_matriz_confusao(y_teste, y_previsto, rotulos)
        st.plotly_chart(figura, use_container_width=True)
        
        # Relatório de classificação
        st.subheader("Relatório de Classificação")
        relatorio = classification_report(y_teste, y_previsto, target_names=rotulos, output_dict=True)
        relatorio_df = pd.DataFrame(relatorio).transpose()
        st.dataframe(relatorio_df)
        
        # Importância das características
        st.subheader("Importância das Características")
        importancia_caracteristicas = pd.DataFrame({
            'caracteristica': caracteristicas,
            'importancia': modelo.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        figura = plotar_importancia_caracteristicas(importancia_caracteristicas)
        st.plotly_chart(figura, use_container_width=True)
        
        # Processar dados de teste se disponíveis
        if arquivo_teste_existe and dados_teste is not None:
            st.header("📤 Previsões para Dados de Teste e Validação pela API")
            
            dados_teste_processados = preprocessar_dados(dados_teste, eh_treino=False, 
                                                       usar_diff_temp=usar_diferenca_temperatura, 
                                                       usar_pot=usar_potencia)
            
            # Garantir que temos as mesmas características
            X_teste_novo = dados_teste_processados[caracteristicas]
            
            # Fazer previsões
            previsoes_teste = modelo.predict(X_teste_novo)
            probabilidades_teste = modelo.predict_proba(X_teste_novo)
            
            # Preparar resultados
            resultados_df = dados_teste.copy()
            
            if tipo_problema == 'binario':
                resultados_df['falha_prevista'] = previsoes_teste
                resultados_df['probabilidade_falha'] = probabilidades_teste[:, 1]
            else:
                resultados_df['tipo_falha_previsto'] = codificador_falha.inverse_transform(previsoes_teste)
                # Adicionar probabilidades para cada classe
                for i, classe in enumerate(codificador_falha.classes_):
                    resultados_df[f'probabilidade_{classe}'] = probabilidades_teste[:, i]
            
            # Exibir resultados
            st.subheader("Previsões para os Dados de Teste")
            st.dataframe(resultados_df.head(10))
            
            # Estatísticas das previsões
            if tipo_problema == 'binario':
                falhas_previstas = resultados_df['falha_prevista'].sum()
                st.metric("Falhas Previstas", f"{falhas_previstas} ({falhas_previstas/len(resultados_df)*100:.1f}%)")
            
            # Botão para download dos resultados
            csv = resultados_df.to_csv(index=False)
            st.download_button(
                label="Download das Previsões (CSV)",
                data=csv,
                file_name="previsoes_teste.csv",
                mime="text/csv"
            )
            
            # Preparar dados para envio à API
            st.subheader("Validação pela API")
            
            # Formatar dados para a API
            dados_para_api = []
            for _, row in resultados_df.iterrows():
                registro = {
                    'id': row['id'],
                    'falha_maquina': int(row['falha_prevista']) if tipo_problema == 'binario' else int(row['tipo_falha_previsto'] != 'NF')
                }
                
                # Adicionar falhas específicas
                if tipo_problema == 'binario':
                    for falha in ['FDF', 'FDC', 'FP', 'FTE', 'FA']:
                        registro[falha] = 0  # Para modelo binário, não sabemos o tipo específico
                else:
                    for falha in ['FDF', 'FDC', 'FP', 'FTE', 'FA']:
                        registro[falha] = int(row['tipo_falha_previsto'] == falha)
                
                dados_para_api.append(registro)
            
            # Botão para enviar para a API
            if st.button("🚀 Validar Modelo na API", type="primary"):
                sucesso, resultado = enviar_para_api(dados_para_api, url_api)
                
                if sucesso:
                    st.success("✅ Validação realizada com sucesso!")
                    exibir_resultados_api(resultado)
                    
                    # Salvar resultados da validação
                    with open('resultado_validacao_api.json', 'w') as f:
                        json.dump(resultado, f, indent=4)
                    
                    st.download_button(
                        label="📥 Download Resultados Validação",
                        data=json.dumps(resultado, indent=4),
                        file_name="resultado_validacao_api.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"❌ Erro na validação: {resultado}")
        
        # Salvar modelo treinado
        if st.button("💾 Salvar Modelo Treinado"):
            joblib.dump(modelo, 'modelo_treinado.pkl')
            st.success("✅ Modelo salvo com sucesso como 'modelo_treinado.pkl'")
            
    else:
        st.error("Erro ao processar os dados de treino.")
else:
    st.error("Arquivo bootcamp_train.csv não encontrado. Por favor, verifique se o arquivo está no diretório correto.")

# Rodapé
st.markdown("---")
st.markdown(
    """
    **Projeto Final do Bootcamp de Ciência de Dados e IA**  
    *Sistema de Manutenção Preditiva para Máquinas Industriais*  
    Desenvolvido com Streamlit, Scikit-learn e Plotly
    """
)

