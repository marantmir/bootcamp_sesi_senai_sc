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
from plotly.subplots import make_subplots
import requests
import io
import joblib

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

# Sidebar para upload de dados e configurações
with st.sidebar:
    st.header("Configurações do Projeto")
    
    # Upload dos dados
    st.subheader("Upload de Dados")
    train_file = st.file_uploader("Dados de Treino (Bootcamp_train.csv)", type="csv")
    test_file = st.file_uploader("Dados de Teste (Bootcamp_test.csv)", type="csv")
    
    # Configurações de análise
    st.subheader("Configurações de Análise")
    eda_option = st.selectbox(
        "Tipo de Análise Exploratória:",
        ["Geral", "Por Tipo de Máquina", "Por Falha"]
    )
    
    # Configurações do modelo
    st.subheader("Configurações do Modelo")
    model_type = st.selectbox(
        "Tipo de Modelagem:",
        ["Problema Binário (Falha vs Não Falha)", "Problema Multiclasse (Tipo de Falha)", "Problema Multirrótulo"]
    )
    
    test_size = st.slider("Percentual de Teste:", 10, 40, 20)
    random_state = st.slider("Random State:", 0, 100, 42)
    
    # API configuration
    st.subheader("Configurações da API")
    api_url = st.text_input("URL da API de avaliação:", "https://api-bootcamp-cdia.herokuapp.com/evaluate")

# Funções auxiliares
@st.cache_data
def load_data(file):
    """Carrega os dados do arquivo CSV"""
    if file is not None:
        return pd.read_csv(file)
    return None

@st.cache_data
def preprocess_data(df, is_train=True):
    """Pré-processa os dados"""
    df_processed = df.copy()
    
    # Codificar tipo de máquina
    le = LabelEncoder()
    df_processed['tipo_encoded'] = le.fit_transform(df_processed['tipo'])
    
    # Calcular diferença de temperatura
    df_processed['diferenca_temperatura'] = df_processed['temperatura_processo'] - df_processed['temperatura_ar']
    
    # Calcular potência (Torque * Velocidade Rotacional)
    df_processed['potencia'] = df_processed['torque'] * df_processed['velocidade_rotacional']
    
    if is_train:
        # Calcular se há qualquer tipo de falha
        falha_cols = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
        df_processed['qualquer_falha'] = df_processed[falha_cols].max(axis=1)
    
    return df_processed

def plot_distributions(df, column, title):
    """Plota distribuições dos dados"""
    fig = px.histogram(df, x=column, title=title, nbins=50)
    fig.update_layout(bargap=0.1)
    return fig

def plot_correlation_matrix(df):
    """Plota matriz de correlação"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlação")
    ))
    
    fig.update_layout(
        title="Matriz de Correlação",
        width=800,
        height=800
    )
    
    return fig

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plota matriz de confusão"""
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm, 
        text_auto=True,
        aspect="auto",
        x=labels,
        y=labels,
        title="Matriz de Confusão"
    )
    fig.update_xaxes(title="Predito")
    fig.update_yaxes(title="Verdadeiro")
    return fig

# Carregar dados
if train_file is not None:
    train_df = load_data(train_file)
    train_processed = preprocess_data(train_df, is_train=True)
    
    # Exibir informações básicas dos dados
    st.header("📊 Análise Exploratória dos Dados")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Amostras", train_df.shape[0])
    with col2:
        st.metric("Total de Features", train_df.shape[1])
    with col3:
        falhas = train_processed['qualquer_falha'].sum()
        st.metric("Máquinas com Falha", f"{falhas} ({falhas/train_df.shape[0]*100:.1f}%)")
    
    # Abas para diferentes análises
    tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Distribuições", "Correlações", "Análise de Falhas"])
    
    with tab1:
        st.subheader("Visualização dos Dados")
        st.dataframe(train_df.head(10))
        
        st.subheader("Informações Estatísticas")
        st.dataframe(train_df.describe())
        
        st.subheader("Tipos de Dados e Valores Nulos")
        info_df = pd.DataFrame({
            'Tipo': train_df.dtypes,
            'Valores Nulos': train_df.isnull().sum(),
            'Percentual Nulos': (train_df.isnull().sum() / len(train_df)) * 100
        })
        st.dataframe(info_df)
    
    with tab2:
        st.subheader("Distribuição das Variáveis Numéricas")
        
        num_cols = ['temperatura_ar', 'temperatura_processo', 'umidade_relativa', 
                   'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta']
        
        for col in num_cols:
            fig = plot_distributions(train_df, col, f"Distribuição de {col}")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Matriz de Correlação")
        fig = plot_correlation_matrix(train_processed)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Análise de Falhas")
        
        # Contagem de falhas por tipo
        falha_cols = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
        falha_counts = train_df[falha_cols].sum()
        
        fig = px.bar(
            x=falha_cols, 
            y=falha_counts.values,
            title="Contagem de Tipos de Falha",
            labels={'x': 'Tipo de Falha', 'y': 'Contagem'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Relação entre variáveis e falhas
        st.subheader("Relação entre Variáveis e Falhas")
        var_option = st.selectbox("Selecione a variável:", num_cols)
        
        fig = px.box(
            train_df, 
            x='falha_maquina', 
            y=var_option,
            title=f"{var_option} vs Falha de Máquina"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Divisão dos dados e treinamento do modelo
    st.header("🤖 Modelagem Preditiva")
    
    # Selecionar features e target baseado no tipo de modelagem
    features = ['tipo_encoded', 'temperatura_ar', 'temperatura_processo', 'umidade_relativa',
               'velocidade_rotacional', 'torque', 'desgaste_da_ferramenta', 
               'diferenca_temperatura', 'potencia']
    
    if model_type == "Problema Binário (Falha vs Não Falha)":
        target = 'qualquer_falha'
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        problem_type = 'binario'
    elif model_type == "Problema Multiclasse (Tipo de Falha)":
        # Criar uma coluna com o tipo de falha predominante
        falha_cols = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
        train_processed['tipo_falha'] = train_processed[falha_cols].idxmax(axis=1)
        # Se não há falha, marca como 'NF'
        train_processed.loc[train_processed['qualquer_falha'] == 0, 'tipo_falha'] = 'NF'
        
        le_falha = LabelEncoder()
        train_processed['tipo_falha_encoded'] = le_falha.fit_transform(train_processed['tipo_falha'])
        
        target = 'tipo_falha_encoded'
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        problem_type = 'multiclasse'
    else:  # Multirrótulo
        target = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
        # Para multirrótulo, precisaríamos de um classificador diferente
        # Por simplicidade, vamos usar um para cada classe
        st.info("Para problemas multirrótulo, treinaremos um modelo para cada tipo de falha.")
        problem_type = 'multirrotulo'
        models = {}
    
    # Dividir dados em treino e teste
    if problem_type != 'multirrotulo':
        X = train_processed[features]
        y = train_processed[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state, stratify=y
        )
        
        # Treinar modelo
        model.fit(X_train, y_train)
        
        # Fazer previsões
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Avaliar modelo
        accuracy = accuracy_score(y_test, y_pred)
        
        st.subheader("Resultados do Modelo")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Acurácia do Modelo", f"{accuracy:.4f}")
        
        with col2:
            st.metric("Amostras de Teste", X_test.shape[0])
        
        # Matriz de confusão
        if problem_type == 'binario':
            labels = ['Sem Falha', 'Com Falha']
        else:
            labels = list(le_falha.classes_)
        
        fig = plot_confusion_matrix(y_test, y_pred, labels)
        st.plotly_chart(fig, use_container_width=True)
        
        # Relatório de classificação
        st.subheader("Relatório de Classificação")
        report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        # Importância das features
        st.subheader("Importância das Features")
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(
            feature_importance, 
            x='importance', 
            y='feature',
            title='Importância das Features no Modelo',
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Processar dados de teste se disponíveis
    if test_file is not None:
        st.header("📤 Previsões para Dados de Teste")
        
        test_df = load_data(test_file)
        test_processed = preprocess_data(test_df, is_train=False)
        
        # Garantir que temos as mesmas features
        X_test_new = test_processed[features]
        
        if problem_type != 'multirrotulo':
            # Fazer previsões
            test_predictions = model.predict(X_test_new)
            test_probabilities = model.predict_proba(X_test_new)
            
            # Preparar resultados
            results_df = test_df.copy()
            
            if problem_type == 'binario':
                results_df['falha_predita'] = test_predictions
                results_df['probabilidade_falha'] = test_probabilities[:, 1]
            else:
                results_df['tipo_falha_predito'] = le_falha.inverse_transform(test_predictions)
                # Adicionar probabilidades para cada classe
                for i, classe in enumerate(le_falha.classes_):
                    results_df[f'probabilidade_{classe}'] = test_probabilities[:, i]
            
            # Exibir resultados
            st.subheader("Previsões para os Dados de Teste")
            st.dataframe(results_df)
            
            # Botão para download dos resultados
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download das Previsões (CSV)",
                data=csv,
                file_name="previsoes_test.csv",
                mime="text/csv"
            )
            
            # Botão para enviar para a API
            if st.button("Enviar Previsões para API de Avaliação"):
                # Preparar dados no formato esperado pela API
                submission_data = results_df[['id']].copy()
                
                if problem_type == 'binario':
                    submission_data['falha_maquina'] = results_df['falha_predita']
                    # Para a API, precisamos preencher as colunas de falha específicas
                    # Como não sabemos o tipo, vamos definir todas como 0
                    for col in ['FDF', 'FDC', 'FP', 'FTE', 'FA']:
                        submission_data[col] = 0
                else:
                    submission_data['falha_maquina'] = (results_df['tipo_falha_predito'] != 'NF').astype(int)
                    # Preencher as colunas de falha específicas
                    for col in ['FDF', 'FDC', 'FP', 'FTE', 'FA']:
                        submission_data[col] = (results_df['tipo_falha_predito'] == col).astype(int)
                
                # Converter para formato JSON
                json_data = submission_data.to_dict(orient='records')
                
                # Enviar para a API
                try:
                    response = requests.post(api_url, json=json_data)
                    if response.status_code == 200:
                        st.success("Previsões enviadas com sucesso para a API!")
                        st.json(response.json())
                    else:
                        st.error(f"Erro ao enviar previsões: {response.text}")
                except Exception as e:
                    st.error(f"Erro de conexão: {str(e)}")
        
        else:
            st.info("Para problemas multirrótulo, é necessário implementar a lógica específica.")
    
    # Salvar modelo treinado
    if st.button("Salvar Modelo Treinado"):
        joblib.dump(model, 'modelo_treinado.pkl')
        st.success("Modelo salvo com sucesso como 'modelo_treinado.pkl'")
        
else:
    st.info("Por favor, faça upload do arquivo de treino para começar a análise.")

# Rodapé
st.markdown("---")
st.markdown(
    """
    **Projeto Final do Bootcamp de Ciência de Dados e IA**  
    Desenvolvido para sistema de manutenção preditiva industrial
    """
)
