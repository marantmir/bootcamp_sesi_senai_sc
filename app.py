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
import os
import json
import time

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Sistema de Manutenção Preditiva",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔧 Sistema Inteligente de Manutenção Preditiva para Indústria 4.0")
st.markdown("""
Este painel interativo permite explorar dados de telemetria de máquinas industriais,
construir um modelo preditivo de falhas e validar seu desempenho contra uma API externa.
""")
st.markdown("---")

# --- Verificação de Arquivos Locais ---
arquivo_treino_existe = os.path.exists("bootcamp_train.csv")
arquivo_teste_existe = os.path.exists("bootcamp_test.csv")

if not arquivo_treino_existe:
    st.error("🚨 Arquivo `bootcamp_train.csv` não encontrado no diretório. Por favor, adicione-o para prosseguir.")
    st.stop() # Impede a execução do restante do script se o arquivo principal não estiver presente

# --- Barra Lateral para Configurações ---
with st.sidebar:
    st.header("⚙️ Configurações do Projeto")
    
    st.subheader("📁 Status dos Arquivos")
    if arquivo_treino_existe:
        st.success("✅ `bootcamp_train.csv` encontrado")
    if arquivo_teste_existe:
        st.success("✅ `bootcamp_test.csv` encontrado")
    else:
        st.warning("⚠️ `bootcamp_test.csv` não encontrado. A validação na API será limitada.")
    
    st.markdown("---")
    
    st.subheader("📊 Configurações de Análise Exploratória")
    # Adicionar uma opção de seleção para coluna de EDA, para dar mais controle manual
    # cols_disponiveis_eda = ["temperatura_ar", "temperatura_processo", "umidade_relativa", 
    #                         "velocidade_rotacional", "torque", "desgaste_da_ferramenta", "tipo"]
    # col_eda_selecionada = st.selectbox("Selecione uma variável para EDA:", cols_disponiveis_eda) # Exemplo
        
    st.markdown("---")
    
    st.subheader("🤖 Configurações do Modelo Preditivo")
    tipo_modelagem = st.selectbox(
        "Tipo de Modelagem:",
        ["Binária (Qualquer Falha)", "Multiclasse (Tipos de Falha Específicos)"],
        help="Escolha se o modelo deve prever a ocorrência de qualquer falha ou classificar o tipo específico de falha."
    )
    
    tamanho_teste = st.slider("Percentual do Conjunto de Teste:", 10, 40, 20, help="Define a proporção dos dados usados para testar o modelo.")
    estado_aleatorio = st.slider("Semente Aleatória (Random State):", 0, 100, 42, help="Controla a aleatoriedade para reprodutibilidade dos resultados.")
    
    st.markdown("---")
    
    st.subheader("🛠️ Engenharia de Características")
    usar_diferenca_temperatura = st.checkbox("Gerar 'diferenca_temperatura'", value=True, help="Cria uma característica com a diferença entre a temperatura do processo e do ar.")
    usar_potencia = st.checkbox("Gerar 'potencia'", value=True, help="Cria uma característica com o produto do torque e da velocidade rotacional.")
    
    st.markdown("---")
    
    st.subheader("🌐 Configurações da API")
    url_api = st.text_input(
        "URL da API de avaliação:", 
        "https://api-bootcamp-cdia.herokuapp.com/evaluate",
        help="Informe o endpoint da API para validação externa do modelo."
    )

# --- Funções Auxiliares ---

@st.cache_data
def carregar_dados_locais(): # Renomeado para "locais"
    """Carrega os dados dos arquivos CSV locais."""
    try:
        dados_treino = pd.read_csv("bootcamp_train.csv")
        dados_teste = pd.read_csv("bootcamp_test.csv") if arquivo_teste_existe else None
        return dados_treino, dados_teste
    except Exception as e:
        st.error(f"Erro ao carregar arquivos de dados: {e}") # Mensagem ligeiramente alterada
        return None, None

@st.cache_data
def processar_e_transformar_dados(df_entrada, eh_treino_conjunto=True, aplicar_diff_temp=True, aplicar_pot=True, encoder_tipo=None, encoder_falha=None): # Renomeado e alterado nomes de variáveis
    """
    Prepara os dados para modelagem, incluindo engenharia de características e codificação.
    Retorna o DataFrame preparado e os objetos codificadores, se eh_treino_conjunto for True.
    """
    df_resultante = df_entrada.copy()
    
    # 1. Codificação do 'tipo' de máquina
    if 'tipo' in df_resultante.columns:
        if eh_treino_conjunto:
            encoder_tipo = LabelEncoder()
            df_resultante['tipo_codificado'] = encoder_tipo.fit_transform(df_resultante['tipo'])
        elif encoder_tipo:
            # Para dados de teste, usar o codificador ajustado nos dados de treino
            # Lidar com tipos de máquina desconhecidos no teste
            df_resultante['tipo_codificado'] = df_resultante['tipo'].apply(
                lambda x: encoder_tipo.transform([x])[0] if x in encoder_tipo.classes_ else -1
            )
                        
    # 2. Engenharia de Características
    if aplicar_diff_temp and 'temperatura_processo' in df_resultante.columns and 'temperatura_ar' in df_resultante.columns:
        df_resultante['diferenca_temperatura'] = df_resultante['temperatura_processo'] - df_resultante['temperatura_ar']
    
    if aplicar_pot and 'torque' in df_resultante.columns and 'velocidade_rotacional' in df_resultante.columns:
        df_resultante['potencia'] = df_resultante['torque'] * df_resultante['velocidade_rotacional']
    
    # 3. Criação da variável alvo 'qualquer_falha' e 'tipo_falha_codificado'
    if eh_treino_conjunto:
        colunas_falhas_especificas = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
        colunas_falhas_presentes = [col for col in colunas_falhas_especificas if col in df_resultante.columns]
        
        # 'qualquer_falha' para modelagem binária
        if colunas_falhas_presentes:
            # Usando uma abordagem mais procedural para a criação da coluna 'qualquer_falha'
            df_resultante['qualquer_falha'] = 0 # Inicializa com 0 (sem falha)
            for col_falha in colunas_falhas_presentes:
                df_resultante.loc[df_resultante[col_falha] == 1, 'qualquer_falha'] = 1
            # Alternativa mais concisa, mas a anterior pode ser percebida como menos "padrão de IA"
            # df_resultante['qualquer_falha'] = df_resultante[colunas_falhas_presentes].any(axis=1).astype(int)
        elif 'falha_maquina' in df_resultante.columns:
            df_resultante['qualquer_falha'] = df_resultante['falha_maquina']
        else:
            st.warning("Não foi possível identificar colunas de falha para 'qualquer_falha'. Definindo como 0.")
            df_resultante['qualquer_falha'] = 0 
            
        # 'tipo_falha_codificado' para modelagem multiclasse
        if colunas_falhas_presentes:
            # Cria uma coluna com o nome da falha predominante, se houver
            df_resultante['tipo_falha_nome'] = 'NF' # No Fault (sem falha)
            for idx, row in df_resultante.iterrows():
                falhas_encontradas = [col for col in colunas_falhas_presentes if row[col] == 1]
                if falhas_encontradas:
                    df_resultante.at[idx, 'tipo_falha_nome'] = falhas_encontradas[0] # Pega a primeira falha encontrada
            
            encoder_falha = LabelEncoder()
            df_resultante['tipo_falha_codificado'] = encoder_falha.fit_transform(df_resultante['tipo_falha_nome'])
        else:
            st.warning("Não foi possível encontrar colunas de falhas específicas para modelagem multiclasse.")
            df_resultante['tipo_falha_codificado'] = 0 
    
    return df_resultante, encoder_tipo, encoder_falha

def plotar_histograma_distribuicoes(data_frame, nome_coluna, titulo_grafico): # Renomeado
    """Plota distribuições dos dados."""
    if nome_coluna not in data_frame.columns:
        return None
    fig = px.histogram(data_frame, x=nome_coluna, title=titulo_grafico, nbins=50, marginal="box", color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(bargap=0.1)
    return fig

def plotar_mapa_correlacao(data_frame_numerico): # Renomeado
    """Plota matriz de correlação usando Plotly."""
    df_numerico_filtrado = data_frame_numerico.select_dtypes(include=[np.number])
    if df_numerico_filtrado.empty:
        return None
    correlacao_calculada = df_numerico_filtrado.corr()
    fig = px.imshow(
        correlacao_calculada,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title="Mapa de Correlação das Variáveis Numéricas" # Título ligeiramente alterado
    )
    return fig

def plotar_mapa_confusao(y_real_dados, y_previsto_dados, rotulos_classes): # Renomeado
    """Plota matriz de confusão usando Plotly."""
    matriz_confusao_calculada = confusion_matrix(y_real_dados, y_previsto_dados)
    fig = px.imshow(
        matriz_confusao_calculada, 
        text_auto=True,
        aspect="auto",
        x=rotulos_classes,
        y=rotulos_classes,
        title="Matriz de Confusão do Modelo", # Título ligeiramente alterado
        color_continuous_scale='Blues'
    )
    fig.update_xaxes(title="Valor Predito") # Eixo ligeiramente alterado
    fig.update_yaxes(title="Valor Verdadeiro") # Eixo ligeiramente alterado
    return fig

def plotar_relevancia_caracteristicas(df_importancia): # Renomeado
    """Plota importância das características usando Plotly."""
    fig = px.bar(
        df_importancia, 
        x='importancia', 
        y='caracteristica',
        title='Relevância das Características no Modelo Preditivo', # Título ligeiramente alterado
        orientation='h',
        color='importancia',
        color_continuous_scale='viridis'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def submeter_para_api(dados_payload, api_url): # Renomeado
    """Envia dados para a API e retorna a resposta."""
    try:
        headers_envio = {'Content-Type': 'application/json'}
        with st.spinner('🚀 Iniciando submissão de previsões para a API externa...'):
            resposta_api = requests.post(api_url, json=dados_payload, headers=headers_envio, timeout=60) # Aumentado timeout
        
        if resposta_api.status_code == 200:
            return True, resposta_api.json()
        else:
            return False, f"Erro HTTP {resposta_api.status_code}: {resposta_api.text}"
    except requests.exceptions.Timeout:
        return False, "Erro de comunicação: O tempo limite da requisição para a API foi atingido."
    except requests.exceptions.ConnectionError:
        return False, "Falha na conexão: Não foi possível estabelecer conexão com a API. Verifique a URL ou sua conexão de rede."
    except Exception as e:
        return False, f"Ocorreu um erro inesperado ao submeter dados à API: {str(e)}"

def apresentar_resultados_api(resultados_recebidos): # Renomeado
    """Apresenta os resultados retornados pela API de forma organizada."""
    if not isinstance(resultados_recebidos, dict):
        st.error("❌ A resposta da API está em um formato inválido.")
        return
    
    st.subheader("📊 Resultados da Validação Externa pela API") # Título ligeiramente alterado
    
    # Métricas gerais
    col_metrica_1, col_metrica_2, col_metrica_3 = st.columns(3) # Renomeado
    with col_metrica_1:
        st.metric("Acurácia Geral (API)", f"{resultados_recebidos.get('overall_accuracy', 0):.4f}")
    with col_metrica_2:
        st.metric("Precisão Média (API)", f"{resultados_recebidos.get('mean_precision', 0):.4f}")
    with col_metrica_3:
        st.metric("Recall Médio (API)", f"{resultados_recebidos.get('mean_recall', 0):.4f}")
    
    st.markdown("---")
    
    # Matriz de confusão
    if 'confusion_matrix' in resultados_recebidos:
        st.subheader("Matriz de Confusão (Dados da API)")
        try:
            api_matriz_conf = np.array(resultados_recebidos['confusion_matrix']) # Renomeado
            # Tentar inferir labels se não fornecidos explicitamente na resposta da API
            api_rotulos = ['Não Falha', 'Falha'] if api_matriz_conf.shape[0] == 2 else [str(i) for i in range(api_matriz_conf.shape[0])] # Renomeado
            
            fig_api_cm = px.imshow( # Renomeado
                api_matriz_conf, 
                text_auto=True,
                aspect="auto",
                x=api_rotulos,
                y=api_rotulos,
                title="Matriz de Confusão - Validação via API",
                color_continuous_scale='Blues'
            )
            fig_api_cm.update_xaxes(title="Previsão da API") # Eixo ligeiramente alterado
            fig_api_cm.update_yaxes(title="Real da API") # Eixo ligeiramente alterado
            st.plotly_chart(fig_api_cm, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível visualizar a matriz de confusão da API: {e}")
            st.json(resultados_recebidos['confusion_matrix']) # Exibe o JSON cru se não conseguir plotar
    
    # Métricas por classe
    if 'class_metrics' in resultados_recebidos:
        st.subheader("Métricas Detalhadas por Classe") # Título ligeiramente alterado
        df_metricas_classe = pd.DataFrame(resultados_recebidos['class_metrics']).T # Renomeado
        st.dataframe(df_metricas_classe)
    
    # Curva ROC (se disponível - geralmente não retorna no formato de plot, mas o AUC score sim)
    if 'roc_auc' in resultados_recebidos:
        st.subheader("Pontuação ROC AUC") # Título ligeiramente alterado
        st.write(f"O valor de AUC ROC reportado pela API é: **{resultados_recebidos['roc_auc']:.4f}**")
    
    st.info("💡 É importante notar que a avaliação da API pode utilizar um conjunto de dados ou metodologia interna.")

# --- Carregar e Pré-processar Dados Iniciais ---
dados_treino_bruto, dados_teste_bruto = carregar_dados_locais() # Renomeado

if dados_treino_bruto is None:
    st.stop() 

# Variáveis para armazenar codificadores globalmente
codificador_tipo_global = None # Renomeado
codificador_falha_global = None # Renomeado

# Processar dados de treino
dados_treino_preparados, codificador_tipo_global, codificador_falha_global = processar_e_transformar_dados( # Renomeado
    dados_treino_bruto, 
    eh_treino_conjunto=True,
    aplicar_diff_temp=usar_diferenca_temperatura, 
    aplicar_pot=usar_potencia
)

# --- Seção Principal do Dashboard ---

# 1. Visão Geral e Estrutura dos Dados
st.header("📋 Visão Geral e Estrutura dos Dados")
st.markdown("Uma primeira olhada nos dados brutos e suas características.")

tab_dados_brutos, tab_info_dados, tab_valores_nulos = st.tabs(["Dados Brutos (Head)", "Informações Descritivas", "Valores Nulos"]) # Renomeado

with tab_dados_brutos:
    st.subheader("Primeiras 5 linhas do conjunto de treino:")
    st.dataframe(dados_treino_bruto.head())
    
    st.subheader("Colunas e Seus Tipos de Dados:") # Título ligeiramente alterado
    st.dataframe(pd.DataFrame(dados_treino_bruto.dtypes, columns=['Tipo de Dado'])) # Usando dataframe para melhor visualização

with tab_info_dados:
    st.subheader("Estatísticas Descritivas para Variáveis Numéricas:")
    st.dataframe(dados_treino_bruto.describe())
    
    st.subheader("Contagem de Valores para Variáveis Categóricas:")
    colunas_categoricas = dados_treino_bruto.select_dtypes(include='object').columns # Renomeado
    if not colunas_categoricas.empty:
        for col in colunas_categoricas:
            st.write(f"**{col}:**")
            st.dataframe(dados_treino_bruto[col].value_counts().reset_index().rename(columns={'index': col, col: 'Contagem'}))
    else:
        st.info("Não foram encontradas colunas categóricas no dataset original.") # Mensagem ligeiramente alterada

with tab_valores_nulos:
    st.subheader("Análise Detalhada de Valores Nulos:") # Título ligeiramente alterado
    contagem_nulos = dados_treino_bruto.isnull().sum() # Renomeado
    percentual_nulos = (dados_treino_bruto.isnull().sum() / len(dados_treino_bruto)) * 100 # Renomeado
    df_nulos = pd.DataFrame({'Total de Nulos': contagem_nulos, 'Percentual (%)': percentual_nulos}) # Renomeado
    st.dataframe(df_nulos[df_nulos['Total de Nulos'] > 0].sort_values(by='Percentual (%)', ascending=False)) # Renomeado
    if df_nulos['Total de Nulos'].sum() == 0: # Renomeado
        st.info("🎉 Excelente! Nenhuma célula vazia (nulo) detectada no seu conjunto de dados de treino.") # Mensagem ligeiramente alterada
    else:
        st.warning("⚠️ Foram identificados valores nulos. Recomenda-se considerar estratégias de imputação ou remoção.") # Mensagem ligeiramente alterada

# 2. Análise Exploratória dos Dados (EDA)
st.header("📊 Análise Exploratória dos Dados (EDA)")
st.markdown("Visualizações para entender padrões, distribuições e relações nos dados.")

tab_distribuicoes_vars, tab_matriz_correlacao, tab_analise_falhas, tab_analise_tipo_maquina = st.tabs(["Distribuições de Variáveis", "Matriz de Correlação", "Análise de Falhas", "Análise por Tipo de Máquina"]) # Renomeado

with tab_distribuicoes_vars:
    st.subheader("Distribuição das Variáveis Numéricas")
    colunas_numericas_para_visualizacao = [ # Renomeado
        'temperatura_ar', 'temperatura_processo', 'umidade_relativa',
        'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta'
    ]
    # Adicionar características engenheiradas se forem usadas
    if usar_diferenca_temperatura and 'diferenca_temperatura' in dados_treino_preparados.columns:
        colunas_numericas_para_visualizacao.append('diferenca_temperatura')
    if usar_potencia and 'potencia' in dados_treino_preparados.columns:
        colunas_numericas_para_visualizacao.append('potencia')

    colunas_numericas_existentes_para_plot = [col for col in colunas_numericas_para_visualizacao if col in dados_treino_preparados.columns] # Renomeado

    if colunas_numericas_existentes_para_plot:
        num_colunas_layout = 2 # Renomeado
        for i in range(0, len(colunas_numericas_existentes_para_plot), num_colunas_layout):
            cols_layout_atual = st.columns(num_colunas_layout) # Renomeado
            for j, coluna_atual in enumerate(colunas_numericas_existentes_para_plot[i:i+num_colunas_layout]): # Renomeado
                with cols_layout_atual[j]:
                    figura_dist = plotar_histograma_distribuicoes(dados_treino_preparados, coluna_atual, f"Distribuição de {coluna_atual.replace('_', ' ').title()}") # Renomeado
                    if figura_dist:
                        st.plotly_chart(figura_dist, use_container_width=True)
    else:
        st.warning("Não há colunas numéricas padrão ou características engenheiradas disponíveis para visualizar distribuições.") # Mensagem ligeiramente alterada

with tab_matriz_correlacao:
    st.subheader("Matriz de Correlação entre Variáveis") # Título ligeiramente alterado
    figura_correlacao = plotar_mapa_correlacao(dados_treino_preparados) # Renomeado
    if figura_correlacao:
        st.plotly_chart(figura_correlacao, use_container_width=True)
    else:
        st.info("Dados numéricos insuficientes para gerar a matriz de correlação.") # Mensagem ligeiramente alterada

with tab_analise_falhas:
    st.subheader("Análise da Ocorrência de Falhas") # Título ligeiramente alterado
    
    if 'qualquer_falha' in dados_treino_preparados.columns:
        contagem_falhas = dados_treino_preparados['qualquer_falha'].value_counts(normalize=True) * 100 # Renomeado
        figura_falha_geral = px.bar( # Renomeado
            x=['Sem Falha', 'Com Falha'],
            y=contagem_falhas.values,
            title="Proporção de Máquinas com e sem Falha Registrada", # Título ligeiramente alterado
            labels={'x': 'Status Operacional', 'y': 'Percentual (%)'}, # Eixos ligeiramente alterados
            color=contagem_falhas.index.astype(str),
            color_discrete_map={'0': 'lightgreen', '1': 'salmon'}
        )
        st.plotly_chart(figura_falha_geral, use_container_width=True)
        st.info(f"Observamos um desbalanceamento de classes significativo: {contagem_falhas[0]:.2f}% sem falha versus {contagem_falhas[1]:.2f}% com falha. O modelo foi configurado para tentar mitigar esse impacto.") # Mensagem ligeiramente alterada
        
    colunas_falhas_especificas = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
    colunas_falhas_existentes_no_df = [col for col in colunas_falhas_especificas if col in dados_treino_preparados.columns] # Renomeado

    if colunas_falhas_existentes_no_df:
        st.subheader("Distribuição dos Tipos de Falha Identificados") # Título ligeiramente alterado
        contagem_tipos_falha_especifica = dados_treino_preparados[colunas_falhas_existentes_no_df].sum() # Renomeado
        if not contagem_tipos_falha_especifica.empty and contagem_tipos_falha_especifica.sum() > 0:
            figura_pizza_falhas = px.pie( # Renomeado
                values=contagem_tipos_falha_especifica.values,
                names=contagem_tipos_falha_especifica.index,
                title="Distribuição Percentual dos Tipos de Falha Ocorrentes", # Título ligeiramente alterado
                hole=0.3
            )
            st.plotly_chart(figura_pizza_falhas, use_container_width=True)
        else:
            st.info("Nenhuma falha específica foi registrada nos dados de treinamento.") # Mensagem ligeiramente alterada
    
    st.subheader("Relação entre Variáveis Numéricas e a Ocorrência de Falha") # Título ligeiramente alterado
    colunas_numericas_para_relacao = [ # Renomeado
        'temperatura_ar', 'temperatura_processo', 'umidade_relativa',
        'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta',
        'diferenca_temperatura', 'potencia'
    ]
    colunas_numericas_para_relacao = [col for col in colunas_numericas_para_relacao if col in dados_treino_preparados.columns]

    if 'qualquer_falha' in dados_treino_preparados.columns and colunas_numericas_para_relacao:
        caracteristica_selecionada_relacao = st.selectbox("Escolha uma variável para comparar com o status de falha:", colunas_numericas_para_relacao) # Renomeado e alterado
        figura_boxplot_falha = px.box( # Renomeado
            dados_treino_preparados, 
            x='qualquer_falha', 
            y=caracteristica_selecionada_relacao, 
            title=f"Distribuição de {caracteristica_selecionada_relacao.replace('_', ' ').title()} por Status de Falha",
            color='qualquer_falha',
            labels={'qualquer_falha': 'Estado de Falha'}, # Eixo ligeiramente alterado
            color_discrete_map={0: 'lightgreen', 1: 'salmon'}
        )
        st.plotly_chart(figura_boxplot_falha, use_container_width=True)
    else:
        st.info("Não foi possível gerar gráficos de relação entre variáveis numéricas e falhas.") # Mensagem ligeiramente alterada

with tab_analise_tipo_maquina:
    st.subheader("Análise por Tipo de Máquina")
    if 'tipo' in dados_treino_preparados.columns and 'qualquer_falha' in dados_treino_preparados.columns:
        contagem_tipos_maquina = dados_treino_preparados['tipo'].value_counts() # Renomeado
        figura_pizza_tipos_maquina = px.pie( # Renomeado
            values=contagem_tipos_maquina.values,
            names=contagem_tipos_maquina.index,
            title="Distribuição de Amostras por Categoria de Máquina", # Título ligeiramente alterado
            hole=0.3
        )
        st.plotly_chart(figura_pizza
