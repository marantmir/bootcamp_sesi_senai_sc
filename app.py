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
import sys
import pkg_resources

# --- Instalação de Dependências ---
# Define um arquivo de "flag" para saber se a verificação e instalação já foram feitas na sessão atual
INSTALL_FLAG_FILE = ".deps_installed_flag"

def check_and_install_dependencies():
    required_packages = {
        'scikit-learn': '1.3.2',
        'numpy': '1.24.3',
        'pandas': '2.0.3',
        'streamlit': '1.26.0',
        'plotly': '5.15.0',
        'requests': '2.31.0',
        'seaborn': '0.13.0'
    }

    # Verifica se a flag já existe. Se sim, assume que as deps já foram tratadas.
    if os.path.exists(INSTALL_FLAG_FILE):
        return

    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    missing_or_wrong_version = []
    for pkg_name, pkg_version in required_packages.items():
        if pkg_name not in installed_packages or installed_packages[pkg_name] != pkg_version:
            missing_or_wrong_version.append(f"{pkg_name}=={pkg_version}")

    if missing_or_wrong_version:
        st.warning(f"Instalando/Atualizando dependências: {', '.join(missing_or_wrong_version)}. Isso pode levar alguns instantes.")
        
        # Cria um placeholder para mensagens de instalação
        install_status = st.empty()
        install_status.info("Iniciando instalação...")

        try:
            python = sys.executable
            for req in missing_or_wrong_version:
                install_status.info(f"Instalando {req}...")
                # Captura a saída para depuração se necessário, mas oculta para o usuário
                result = subprocess.run(
                    [python, '-m', 'pip', 'install', '--upgrade', req],
                    capture_output=True, text=True, check=True # check=True levanta exceção se houver erro
                )
                st.code(result.stdout) para ver o log de instalação
                st.error(result.stderr) para ver erros específicos do pip

            st.success("Dependências instaladas/atualizadas com sucesso!")
            # Cria a flag para indicar que a instalação foi concluída
            with open(INSTALL_FLAG_FILE, "w") as f:
                f.write("Dependencies installed.")
            
            time.sleep(1) 
            st.experimental_rerun()
        except subprocess.CalledProcessError as e:
            install_status.error(f"Erro ao instalar '{e.cmd[5]}'. Erro: {e.stderr}. Por favor, tente novamente ou verifique seu ambiente Python.")
            st.stop()
        except Exception as e:
            install_status.error(f"Erro inesperado durante a instalação das dependências: {e}")
            st.stop()
    else:
        # Se tudo estiver ok, cria a flag para evitar verificações futuras na mesma sessão
        with open(INSTALL_FLAG_FILE, "w") as f:
            f.write("Dependencies installed.")

# Chama a função de verificação e instalação no início do script
check_and_install_dependencies()

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

# --- Verificação de Arquivos Locais (Melhoria para feedback visual) ---
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
def carregar_dados():
    """Carrega os dados dos arquivos CSV locais."""
    try:
        dados_treino = pd.read_csv("bootcamp_train.csv")
        dados_teste = pd.read_csv("bootcamp_test.csv") if arquivo_teste_existe else None
        return dados_treino, dados_teste
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        return None, None

@st.cache_data
def preprocessar_dados(df, eh_treino=True, usar_diff_temp=True, usar_pot=True, codificador_tipo=None, codificador_falha=None):
    """
    Pré-processa os dados, aplicando engenharia de características e codificação.
    Retorna o DataFrame processado e os codificadores se eh_treino for True.
    """
    df_processado = df.copy()
    
    # 1. Codificação do 'tipo' de máquina
    if 'tipo' in df_processado.columns:
        if eh_treino:
            codificador_tipo = LabelEncoder()
            df_processado['tipo_codificado'] = codificador_tipo.fit_transform(df_processado['tipo'])
        elif codificador_tipo:
            # Para dados de teste, usar o codificador ajustado nos dados de treino
            # Lidar com tipos de máquina desconhecidos no teste
            df_processado['tipo_codificado'] = df_processado['tipo'].apply(
                lambda x: codificador_tipo.transform([x])[0] if x in codificador_tipo.classes_ else -1
            )
                        
    # 2. Engenharia de Características
    if usar_diff_temp and 'temperatura_processo' in df_processado.columns and 'temperatura_ar' in df_processado.columns:
        df_processado['diferenca_temperatura'] = df_processado['temperatura_processo'] - df_processado['temperatura_ar']
    
    if usar_pot and 'torque' in df_processado.columns and 'velocidade_rotacional' in df_processado.columns:
        df_processado['potencia'] = df_processado['torque'] * df_processado['velocidade_rotacional']
    
    # 3. Criação da variável alvo 'qualquer_falha' e 'tipo_falha_codificado'
    if eh_treino:
        colunas_falhas_especificas = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
        colunas_falhas_existentes = [col for col in colunas_falhas_especificas if col in df_processado.columns]
        
        # 'qualquer_falha' para modelagem binária
        if colunas_falhas_existentes:
            df_processado['qualquer_falha'] = df_processado[colunas_falhas_existentes].max(axis=1)
        elif 'falha_maquina' in df_processado.columns:
            df_processado['qualquer_falha'] = df_processado['falha_maquina']
        else:
            st.warning("Não foi possível encontrar colunas de falha para 'qualquer_falha'.")
            df_processado['qualquer_falha'] = 0 # Default para evitar erros
            
        # 'tipo_falha_codificado' para modelagem multiclasse
        if colunas_falhas_existentes:
            # Cria uma coluna com o nome da falha predominante, se houver
            # Se houver mais de uma falha, idxmax pega a primeira que aparecer na lista de colunas_falhas_existentes
            df_processado['tipo_falha'] = df_processado[colunas_falhas_existentes].apply(
                lambda x: x.idxmax() if x.sum() > 0 else 'NF', axis=1
            )
            codificador_falha = LabelEncoder()
            df_processado['tipo_falha_codificado'] = codificador_falha.fit_transform(df_processado['tipo_falha'])
        else:
            st.warning("Não foi possível encontrar colunas de falhas específicas para modelagem multiclasse.")
            df_processado['tipo_falha_codificado'] = 0 # Default
    
    return df_processado, codificador_tipo, codificador_falha


def plotar_distribuicoes(df, coluna, titulo):
    """Plota distribuições dos dados."""
    if coluna not in df.columns:
        return None
    fig = px.histogram(df, x=coluna, title=titulo, nbins=50, marginal="box", color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(bargap=0.1)
    return fig

def plotar_matriz_correlacao(df):
    """Plota matriz de correlação usando Plotly."""
    df_numerico = df.select_dtypes(include=[np.number])
    if df_numerico.empty:
        return None
    correlacao = df_numerico.corr()
    fig = px.imshow(
        correlacao,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title="Matriz de Correlação das Variáveis Numéricas"
    )
    return fig

def plotar_matriz_confusao(y_real, y_previsto, rotulos):
    """Plota matriz de confusão usando Plotly."""
    matriz_confusao = confusion_matrix(y_real, y_previsto)
    fig = px.imshow(
        matriz_confusao, 
        text_auto=True,
        aspect="auto",
        x=rotulos,
        y=rotulos,
        title="Matriz de Confusão",
        color_continuous_scale='Blues'
    )
    fig.update_xaxes(title="Previsto")
    fig.update_yaxes(title="Real")
    return fig

def plotar_importancia_caracteristicas(importancia_df):
    """Plota importância das características usando Plotly."""
    fig = px.bar(
        importancia_df, 
        x='importancia', 
        y='caracteristica',
        title='Importância das Características no Modelo',
        orientation='h',
        color='importancia',
        color_continuous_scale='viridis'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def enviar_para_api(dados_envio, url_api):
    """Envia dados para a API e retorna a resposta."""
    try:
        headers = {'Content-Type': 'application/json'}
        with st.spinner('🚀 Enviando previsões para a API de avaliação...'):
            resposta = requests.post(url_api, json=dados_envio, headers=headers, timeout=60) # Aumentado timeout
        
        if resposta.status_code == 200:
            return True, resposta.json()
        else:
            return False, f"Erro HTTP {resposta.status_code}: {resposta.text}"
    except requests.exceptions.Timeout:
        return False, "Erro de conexão: O tempo limite da requisição foi excedido."
    except requests.exceptions.ConnectionError:
        return False, "Erro de conexão: Não foi possível conectar à API. Verifique a URL ou sua conexão."
    except Exception as e:
        return False, f"Erro inesperado ao enviar para a API: {str(e)}"

def exibir_resultados_api(resultados):
    """Exibe os resultados retornados pela API de forma organizada."""
    if not isinstance(resultados, dict):
        st.error("❌ Formato de resposta da API inválido.")
        return
    
    st.subheader("📊 Resultados da Validação pela API")
    
    # Métricas gerais
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Acurácia Geral", f"{resultados.get('overall_accuracy', 0):.4f}")
    with col2:
        st.metric("Precision Média", f"{resultados.get('mean_precision', 0):.4f}")
    with col3:
        st.metric("Recall Médio", f"{resultados.get('mean_recall', 0):.4f}")
    
    st.markdown("---")
    
    # Matriz de confusão
    if 'confusion_matrix' in resultados:
        st.subheader("Confusion Matrix (API)")
        try:
            api_conf_matrix = np.array(resultados['confusion_matrix'])
            # Tentar inferir labels se não fornecidos explicitamente na resposta da API
            api_labels = ['Não Falha', 'Falha'] if api_conf_matrix.shape[0] == 2 else [str(i) for i in range(api_conf_matrix.shape[0])]
            
            fig = px.imshow(
                api_conf_matrix, 
                text_auto=True,
                aspect="auto",
                x=api_labels,
                y=api_labels,
                title="Matriz de Confusão - Validação API",
                color_continuous_scale='Blues'
            )
            fig.update_xaxes(title="Previsto")
            fig.update_yaxes(title="Real")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível plotar a matriz de confusão da API: {e}")
            st.json(resultados['confusion_matrix']) # Exibe o JSON cru se não conseguir plotar
    
    # Métricas por classe
    if 'class_metrics' in resultados:
        st.subheader("Métricas por Classe")
        metricas_df = pd.DataFrame(resultados['class_metrics']).T
        st.dataframe(metricas_df)
    
    # Curva ROC (se disponível - geralmente não retorna no formato de plot, mas o AUC score sim)
    if 'roc_auc' in resultados:
        st.subheader("ROC AUC Score")
        st.write(f"O valor de AUC ROC retornado pela API é: **{resultados['roc_auc']:.4f}**")
    
    st.info("💡 Lembre-se que a API pode avaliar o modelo de forma ligeiramente diferente com base em seus dados de teste internos.")

# --- Carregar e Pré-processar Dados Iniciais ---
dados_treino_raw, dados_teste_raw = carregar_dados()

if dados_treino_raw is None:
    st.stop() # Parar se não houver dados de treino para evitar erros

# Variáveis para armazenar codificadores
global_codificador_tipo = None
global_codificador_falha = None

# Processar dados de treino
dados_processados_treino, global_codificador_tipo, global_codificador_falha = preprocessar_dados(
    dados_treino_raw, 
    eh_treino=True,
    usar_diff_temp=usar_diferenca_temperatura, 
    usar_pot=usar_potencia
)

# --- Seção Principal do Dashboard ---

# 1. Visão Geral e Estrutura dos Dados
st.header("📋 Visão Geral e Estrutura dos Dados")
st.markdown("Uma primeira olhada nos dados brutos e suas características.")

tab_raw_data, tab_data_info, tab_null_values = st.tabs(["Dados Brutos (Head)", "Informações Descritivas", "Valores Nulos"])

with tab_raw_data:
    st.subheader("Primeiras 5 linhas do conjunto de treino:")
    st.dataframe(dados_treino_raw.head())
    
    st.subheader("Colunas e Tipos de Dados:")
    st.write(pd.DataFrame(dados_treino_raw.dtypes, columns=['Tipo de Dado']))

with tab_data_info:
    st.subheader("Estatísticas Descritivas para Variáveis Numéricas:")
    st.dataframe(dados_treino_raw.describe())
    
    st.subheader("Contagem de Valores para Variáveis Categóricas:")
    categorical_cols = dados_treino_raw.select_dtypes(include='object').columns
    if not categorical_cols.empty:
        for col in categorical_cols:
            st.write(f"**{col}:**")
            st.dataframe(dados_treino_raw[col].value_counts().reset_index().rename(columns={'index': col, col: 'Contagem'}))
    else:
        st.info("Não há colunas categóricas no dataset original.")

with tab_null_values:
    st.subheader("Análise de Valores Nulos:")
    null_counts = dados_treino_raw.isnull().sum()
    null_percentage = (dados_treino_raw.isnull().sum() / len(dados_treino_raw)) * 100
    null_df = pd.DataFrame({'Valores Nulos': null_counts, 'Percentual': null_percentage})
    st.dataframe(null_df[null_df['Valores Nulos'] > 0].sort_values(by='Percentual', ascending=False))
    if null_df['Valores Nulos'].sum() == 0:
        st.info("🎉 Ótima notícia! Não há valores nulos no seu conjunto de dados de treino.")
    else:
        st.warning("⚠️ Foram encontrados valores nulos. Considere estratégias de tratamento de nulos.")

# 2. Análise Exploratória dos Dados (EDA)
st.header("📊 Análise Exploratória dos Dados (EDA)")
st.markdown("Visualizações para entender padrões, distribuições e relações nos dados.")

eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(["Distribuições de Variáveis", "Matriz de Correlação", "Análise de Falhas", "Análise por Tipo de Máquina"])

with eda_tab1:
    st.subheader("Distribuição das Variáveis Numéricas")
    colunas_numericas_para_plot = [
        'temperatura_ar', 'temperatura_processo', 'umidade_relativa',
        'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta'
    ]
    # Adicionar características engenheiradas se forem usadas
    if usar_diferenca_temperatura and 'diferenca_temperatura' in dados_processados_treino.columns:
        colunas_numericas_para_plot.append('diferenca_temperatura')
    if usar_potencia and 'potencia' in dados_processados_treino.columns:
        colunas_numericas_para_plot.append('potencia')

    colunas_numericas_existentes = [col for col in colunas_numericas_para_plot if col in dados_processados_treino.columns]

    if colunas_numericas_existentes:
        colunas_por_linha = 2
        for i in range(0, len(colunas_numericas_existentes), colunas_por_linha):
            cols_layout = st.columns(colunas_por_linha)
            for j, coluna in enumerate(colunas_numericas_existentes[i:i+colunas_por_linha]):
                with cols_layout[j]:
                    fig = plotar_distribuicoes(dados_processados_treino, coluna, f"Distribuição de {coluna.replace('_', ' ').title()}")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Nenhuma coluna numérica padrão ou engenheirada encontrada para plotar distribuições.")

with eda_tab2:
    st.subheader("Matriz de Correlação")
    fig_corr = plotar_matriz_correlacao(dados_processados_treino)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Não há dados numéricos suficientes para calcular a matriz de correlação.")

with eda_tab3:
    st.subheader("Análise de Falhas")
    
    if 'qualquer_falha' in dados_processados_treino.columns:
        falhas_count = dados_processados_treino['qualquer_falha'].value_counts(normalize=True) * 100
        fig_falha_geral = px.bar(
            x=['Sem Falha', 'Com Falha'],
            y=falhas_count.values,
            title="Percentual de Máquinas com e sem Falha",
            labels={'x': 'Status da Máquina', 'y': 'Percentual (%)'},
            color=falhas_count.index.astype(str),
            color_discrete_map={'0': 'lightgreen', '1': 'salmon'}
        )
        st.plotly_chart(fig_falha_geral, use_container_width=True)
        st.info(f"Observa-se um desbalanceamento de classes: {falhas_count[0]:.2f}% sem falha vs {falhas_count[1]:.2f}% com falha. O modelo está configurado para tentar mitigar isso.")
        
    colunas_falhas_especificas = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
    colunas_falhas_existentes = [col for col in colunas_falhas_especificas if col in dados_processados_treino.columns]

    if colunas_falhas_existentes:
        st.subheader("Distribuição de Tipos de Falha Específicos")
        contagem_tipos_falha = dados_processados_treino[colunas_falhas_existentes].sum()
        if not contagem_tipos_falha.empty and contagem_tipos_falha.sum() > 0:
            fig_pie_falhas = px.pie(
                values=contagem_tipos_falha.values,
                names=contagem_tipos_falha.index,
                title="Distribuição dos Tipos de Falha Ocorrentes",
                hole=0.3
            )
            st.plotly_chart(fig_pie_falhas, use_container_width=True)
        else:
            st.info("Nenhuma falha específica registrada nos dados de treino.")
    
    st.subheader("Relação entre Variáveis Numéricas e Ocorrência de Falha")
    colunas_numericas_eda = [
        'temperatura_ar', 'temperatura_processo', 'umidade_relativa',
        'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta',
        'diferenca_temperatura', 'potencia'
    ]
    colunas_numericas_eda = [col for col in colunas_numericas_eda if col in dados_processados_treino.columns]

    if 'qualquer_falha' in dados_processados_treino.columns and colunas_numericas_eda:
        selected_feature = st.selectbox("Selecione uma variável para comparar com a falha:", colunas_numericas_eda)
        fig_box = px.box(
            dados_processados_treino, 
            x='qualquer_falha', 
            y=selected_feature, 
            title=f"Distribuição de {selected_feature.replace('_', ' ').title()} por Status de Falha",
            color='qualquer_falha',
            labels={'qualquer_falha': 'Status da Máquina'},
            color_discrete_map={0: 'lightgreen', 1: 'salmon'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Não foi possível gerar gráficos de relação entre variáveis e falhas.")

with eda_tab4:
    st.subheader("Análise por Tipo de Máquina")
    if 'tipo' in dados_processados_treino.columns and 'qualquer_falha' in dados_processados_treino.columns:
        contagem_tipos = dados_processados_treino['tipo'].value_counts()
        fig_pie_tipos = px.pie(
            values=contagem_tipos.values,
            names=contagem_tipos.index,
            title="Distribuição de Amostras por Tipo de Máquina",
            hole=0.3
        )
        st.plotly_chart(fig_pie_tipos, use_container_width=True)
        
        falhas_por_tipo = dados_processados_treino.groupby('tipo')['qualquer_falha'].mean().reset_index()
        falhas_por_tipo['Percentual de Falhas'] = falhas_por_tipo['qualquer_falha'] * 100
        fig_bar_falhas_tipo = px.bar(
            falhas_por_tipo, 
            x='tipo', 
            y='Percentual de Falhas', 
            title="Percentual de Falhas por Tipo de Máquina",
            labels={'tipo': 'Tipo de Máquina'},
            color='Percentual de Falhas',
            color_continuous_scale='Plasma'
        )
        st.plotly_chart(fig_bar_falhas_tipo, use_container_width=True)
    else:
        st.info("As colunas 'tipo' ou 'qualquer_falha' não foram encontradas para esta análise.")

# 3. Modelagem Preditiva
st.header("🤖 Modelagem Preditiva")
st.markdown("Construção, treinamento e avaliação do modelo de machine learning.")

# Selecionar características baseado nas opções e disponibilidade
caracteristicas_base = ['temperatura_ar', 'temperatura_processo', 'umidade_relativa',
                       'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta']
caracteristicas_selecionadas = [col for col in caracteristicas_base if col in dados_processados_treino.columns]

if 'tipo_codificado' in dados_processados_treino.columns:
    caracteristicas_selecionadas.insert(0, 'tipo_codificado') # Adiciona no início

if usar_diferenca_temperatura and 'diferenca_temperatura' in dados_processados_treino.columns:
    caracteristicas_selecionadas.append('diferenca_temperatura')
if usar_potencia and 'potencia' in dados_processados_treino.columns:
    caracteristicas_selecionadas.append('potencia')

if not caracteristicas_selecionadas:
    st.error("❌ Nenhuma característica adequada encontrada para modelagem. Verifique os dados e as configurações.")
else:
    st.info(f"Características selecionadas para o modelo: {', '.join(caracteristicas_selecionadas)}")
    
    alvo = None
    modelo = None
    rotulos_alvo = None
    
    if tipo_modelagem == "Binária (Qualquer Falha)":
        if 'qualquer_falha' in dados_processados_treino.columns:
            alvo = 'qualquer_falha'
            modelo = RandomForestClassifier(n_estimators=100, random_state=estado_aleatorio, class_weight='balanced')
            rotulos_alvo = ['Sem Falha', 'Com Falha']
        else:
            st.error("A coluna 'qualquer_falha' não está disponível para modelagem binária.")
            
    elif tipo_modelagem == "Multiclasse (Tipos de Falha Específicos)":
        if 'tipo_falha_codificado' in dados_processados_treino.columns and global_codificador_falha:
            alvo = 'tipo_falha_codificado'
            modelo = RandomForestClassifier(n_estimators=100, random_state=estado_aleatorio, class_weight='balanced')
            rotulos_alvo = global_codificador_falha.classes_
        else:
            st.error("As colunas de tipos de falha ou o codificador não estão disponíveis para modelagem multiclasse.")

    if alvo and alvo in dados_processados_treino.columns and modelo:
        X = dados_processados_treino[caracteristicas_selecionadas]
        y = dados_processados_treino[alvo]
        
        if len(y.unique()) < 2:
            st.error(f"Não há variação suficiente na variável alvo '{alvo}' para treinamento do modelo. Necessário pelo menos 2 classes distintas.")
        else:
            try:
                # Divisão dos dados em treino e teste
                X_treino, X_teste, y_treino, y_teste = train_test_split(
                    X, y, test_size=tamanho_teste/100, random_state=estado_aleatorio, stratify=y
                )
                
                # Treinar modelo
                with st.spinner('Treinando modelo RandomForest...'):
                    modelo.fit(X_treino, y_treino)
                st.success("Modelo treinado com sucesso!")
                
                # Fazer previsões
                y_previsto = modelo.predict(X_teste)
                y_probabilidade = modelo.predict_proba(X_teste)
                
                st.subheader("Resultados do Modelo (Validação Interna)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Acurácia", f"{accuracy_score(y_teste, y_previsto):.4f}")
                with col2:
                    st.metric("Amostras de Treino", X_treino.shape[0])
                with col3:
                    st.metric("Amostras de Teste", X_teste.shape[0])
                
                # Matriz de Confusão
                if rotulos_alvo is not None:
                    fig_cm = plotar_matriz_confusao(y_teste, y_previsto
