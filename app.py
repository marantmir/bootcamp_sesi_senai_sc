import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÃO INICIAL DO STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Sistema de Manutenção Preditiva - Bootcamp CDIA", 
    page_icon="🔧", 
    layout="wide"
)

st.title("🔧 Sistema Inteligente de Manutenção Preditiva Avançado")
st.markdown("### Bootcamp de Ciência de Dados e IA - Projeto Final com Melhorias")

# =============================================================================
# SEÇÃO 1: ENTENDIMENTO DO PROBLEMA
# =============================================================================
st.header("📋 1. ENTENDIMENTO DO PROBLEMA")

with st.expander("🎯 Contexto e Objetivos", expanded=True):
    st.markdown("""
    **Problema de Negócio:**
    - Empresa industrial necessita de sistema inteligente para manutenção preditiva
    - Dados coletados via dispositivos IoT de diferentes máquinas
    - Objetivo: Identificar falhas antes que ocorram e classificar o tipo de falha
    
    **Impacto Esperado:**
    - ⚡ Redução de paradas não programadas (até 30%)
    - 💰 Economia em custos de manutenção (até 25%)
    - 📈 Aumento da eficiência operacional (até 15%)
    - 🔒 Melhoria na segurança industrial
    
    **Tipos de Falhas Monitoradas:**
    - **FDF**: Falha por Desgaste da Ferramenta
    - **FDC**: Falha por Dissipação de Calor  
    - **FP**: Falha de Potência
    - **FTE**: Falha por Tensão Excessiva
    - **FA**: Falha Aleatória
    
    **Metodologia Avançada Aplicada:**
    - 🤖 Múltiplos algoritmos de ML (RF, XGBoost, LightGBM, Neural Networks)
    - 🔧 Engenharia avançada de features
    - ⚙️ Otimização de hiperparâmetros
    - 🎯 Ensemble de modelos
    """)

# =============================================================================
# CONSTANTES E CONFIGURAÇÕES
# =============================================================================
COLS_FALHA = ["fdf", "fdc", "fp", "fte", "fa"]
COL_ALVO_BINARIA = "falha_maquina"

# =============================================================================
# SEÇÃO 2: CARREGAMENTO E CONFIGURAÇÃO AVANÇADA
# =============================================================================
st.header("📁 2. CONFIGURAÇÃO AVANÇADA DO SISTEMA")

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações Avançadas")
    
    # Upload de arquivos
    arquivo_treino = st.file_uploader("📊 Bootcamp_train.csv", type=["csv"])
    arquivo_teste = st.file_uploader("🎯 Bootcamp_test.csv (opcional)", type=["csv"])
    
    st.subheader("🔧 Parâmetros de Modelagem")
    tipo_modelagem = st.selectbox(
        "Abordagem:", 
        ["Multirrótulo", "Binária", "Multiclasse"],
        help="Multirrótulo: prediz múltiplas falhas simultâneas"
    )
    
    algoritmo = st.selectbox(
        "Algoritmo Principal:",
        ["Ensemble (Todos)", "Random Forest", "XGBoost", "LightGBM", "Neural Network"],
        help="Ensemble combina todos os algoritmos"
    )
    
    otimizar_hiper = st.checkbox("🔍 Otimização de Hiperparâmetros", value=False)
    tipo_busca = st.selectbox("Tipo de Busca:", ["RandomizedSearch", "GridSearch"]) if otimizar_hiper else "Nenhuma"
    
    percentual_teste = st.slider("% Validação", 10, 40, 20, 5)
    semente = st.slider("Semente Aleatória", 0, 9999, 42)
    
    st.subheader("🔬 Engenharia Avançada de Features")
    usar_dif_temp = st.checkbox("Diferença de temperatura", value=True)
    usar_potencia = st.checkbox("Potência (torque×velocidade)", value=True)
    usar_eficiencia = st.checkbox("Eficiência da ferramenta", value=True)
    usar_interacoes = st.checkbox("🔗 Interações entre variáveis", value=True)
    usar_series_temporais = st.checkbox("⏰ Features temporais", value=True)
    
    st.subheader("🎯 Seleção de Features")
    usar_selecao = st.checkbox("Seleção automática de features", value=True)
    n_features = st.slider("Número de features (se seleção ativa)", 10, 50, 20) if usar_selecao else None
    
    normalizar = st.checkbox("Normalização (StandardScaler)", value=True)

# =============================================================================
# FUNÇÕES UTILITÁRIAS AVANÇADAS
# =============================================================================
@st.cache_data
def carregar_e_processar_dados(arquivo):
    """Carrega e processa os dados básicos"""
    df = pd.read_csv(arquivo)
    
    # Normalizar nomes das colunas
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    )
    
    return df

def criar_features_avancadas(df):
    """Aplica engenharia avançada de features"""
    df = df.copy()
    
    # Features básicas de engenharia
    if usar_dif_temp and {"temperatura_processo","temperatura_ar"}.issubset(df.columns):
        df["dif_temperatura"] = df["temperatura_processo"] - df["temperatura_ar"]
        df["razao_temperatura"] = df["temperatura_processo"] / (df["temperatura_ar"] + 1e-6)
        df["temp_normalizada"] = (df["temperatura_processo"] - df["temperatura_ar"].mean()) / df["temperatura_ar"].std()
    
    if usar_potencia and {"torque","velocidade_rotacional"}.issubset(df.columns):
        df["potencia_kw"] = (df["torque"] * df["velocidade_rotacional"]) / 1000.0
        df["potencia_especifica"] = df["potencia_kw"] / (df["desgaste_da_ferramenta"] + 1.0)
        df["torque_por_velocidade"] = df["torque"] / (df["velocidade_rotacional"] + 1.0)
    
    if usar_eficiencia and {"torque","desgaste_da_ferramenta"}.issubset(df.columns):
        df["eficiencia_ferramenta"] = df["torque"] / (df["desgaste_da_ferramenta"] + 1.0)
        df["desgaste_normalizado"] = df["desgaste_da_ferramenta"] / df["torque"]
        df["stress_ferramenta"] = df["torque"] * df["desgaste_da_ferramenta"]
    
    # Features baseadas em séries temporais (simuladas através do índice)
    if usar_series_temporais:
        # Rolling statistics (usando janela baseada no índice)
        window_size = min(10, len(df) // 4)
        if window_size > 2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in ['temperatura_processo', 'torque', 'velocidade_rotacional']:
                if col in numeric_cols:
                    df[f"{col}_media_movel"] = df[col].rolling(window=window_size, min_periods=1).mean()
                    df[f"{col}_desvio_movel"] = df[col].rolling(window=window_size, min_periods=1).std().fillna(0)
                    df[f"{col}_tendencia"] = df[col].diff().fillna(0)
    
    # Features de interação entre variáveis
    if usar_interacoes:
        # Interações importantes identificadas
        if all(col in df.columns for col in ["temperatura_processo", "umidade_relativa"]):
            df["temp_umidade_interacao"] = df["temperatura_processo"] * df["umidade_relativa"]
        
        if all(col in df.columns for col in ["torque", "umidade_relativa"]):
            df["torque_umidade_interacao"] = df["torque"] * df["umidade_relativa"]
            
        if all(col in df.columns for col in ["velocidade_rotacional", "temperatura_processo"]):
            df["velocidade_temp_interacao"] = df["velocidade_rotacional"] * df["temperatura_processo"]
    
    return df

def otimizar_hiperparametros(modelo, X, y, algoritmo_nome, tipo_busca="RandomizedSearch"):
    """Otimiza hiperparâmetros do modelo"""
    
    param_grids = {
        "Random Forest": {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        "XGBoost": {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        },
        "LightGBM": {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100]
        },
        "Neural Network": {
            'hidden_layer_sizes': [(50,), (100,), (50, 30)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['adaptive', 'constant']
        }
    }
    
    if algoritmo_nome not in param_grids:
        return modelo
    
    param_grid = param_grids[algoritmo_nome]
    
    if tipo_busca == "GridSearch":
        search = GridSearchCV(modelo, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    else:
        search = RandomizedSearchCV(modelo, param_grid, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
    
    search.fit(X, y)
    return search.best_estimator_

def criar_ensemble(X_train, y_train, tipo_modelagem):
    """Cria ensemble de modelos"""
    modelos_base = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgb': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'lgb': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'nn': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }
    
    if tipo_modelagem == "Multirrótulo":
        ensemble_models = {}
        for name, model in modelos_base.items():
            ensemble_models[name] = MultiOutputClassifier(model)
        
        # Para multirrótulo, retornamos um dicionário de modelos
        for name, model in ensemble_models.items():
            model.fit(X_train, y_train)
        
        return ensemble_models
    else:
        # Para binário/multiclasse, usamos VotingClassifier
        ensemble = VotingClassifier(
            estimators=list(modelos_base.items()),
            voting='soft'
        )
        ensemble.fit(X_train, y_train)
        return ensemble

# Carregar dados
dados_treino = None
dados_teste = None

if arquivo_treino:
    dados_treino = carregar_e_processar_dados(arquivo_treino)
    st.success(f"✅ Dados de treino carregados: {dados_treino.shape[0]} amostras × {dados_treino.shape[1]} features")
    
if arquivo_teste:
    dados_teste = carregar_e_processar_dados(arquivo_teste)
    st.success(f"✅ Dados de teste carregados: {dados_teste.shape[0]} amostras")

if dados_treino is None:
    st.warning("⏳ Por favor, carregue o arquivo de treino para continuar.")
    st.stop()

# =============================================================================
# SEÇÃO 3: ANÁLISE EXPLORATÓRIA AVANÇADA
# =============================================================================
st.header("📊 3. ANÁLISE EXPLORATÓRIA AVANÇADA")

# Aplicar engenharia de features
dados_processados = criar_features_avancadas(dados_treino)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total de Amostras", dados_processados.shape[0])
with col2:
    st.metric("📈 Features Originais", dados_treino.shape[1])
with col3:
    st.metric("🔧 Features Criadas", dados_processados.shape[1] - dados_treino.shape[1])
with col4:
    st.metric("📊 Total de Features", dados_processados.shape[1])

if all(col in dados_processados.columns for col in COLS_FALHA):
    # Análise de desbalanceamento avançada
    st.subheader("⚖️ Análise Detalhada de Desbalanceamento")
    
    falhas_count = dados_processados[COLS_FALHA].sum().sort_values(ascending=False)
    total_amostras = len(dados_processados)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tabela de estatísticas
        df_stats = pd.DataFrame({
            'Tipo_Falha': falhas_count.index,
            'Quantidade': falhas_count.values,
            'Percentual': (falhas_count.values / total_amostras * 100).round(2),
            'Classe': ['Minoritária' if x < total_amostras * 0.1 else 'Majoritária' for x in falhas_count.values]
        })
        
        st.dataframe(df_stats, use_container_width=True)
        
        # Calcular razão de desbalanceamento
        ratio_desbalanceamento = falhas_count.max() / falhas_count.min()
        st.metric("🔄 Razão de Desbalanceamento", f"{ratio_desbalanceamento:.1f}:1")
    
    with col2:
        # Gráfico de distribuição
        fig = px.bar(
            df_stats, 
            x='Quantidade', 
            y='Tipo_Falha',
            color='Classe',
            orientation='h',
            title="Distribuição das Classes de Falhas"
        )
        st.plotly_chart(fig, use_container_width=True)

# Análise de correlações das novas features
st.subheader("🔗 Matriz de Correlação das Features Avançadas")

# Selecionar apenas features numéricas criadas recentemente
new_features = [col for col in dados_processados.columns 
                if any(keyword in col for keyword in ['interacao', 'media_movel', 'desvio_movel', 'tendencia', 
                                                     'potencia', 'eficiencia', 'dif_', 'razao_'])]

if new_features:
    correlation_matrix = dados_processados[new_features].corr()
    
    fig_corr = px.imshow(
        correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        aspect="auto",
        title="Correlação entre Features Criadas",
        color_continuous_scale="RdBu_r"
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

# =============================================================================
# SEÇÃO 4: PREPARAÇÃO AVANÇADA DOS DADOS
# =============================================================================
st.header("🔧 4. PREPARAÇÃO AVANÇADA DOS DADOS")

# Preparar features (X) e targets (y)
excluir_cols = {"id", "id_produto", "tipo", COL_ALVO_BINARIA} | set(COLS_FALHA)
features_cols = [col for col in dados_processados.columns if col not in excluir_cols]

# Tratar valores categóricos
if "tipo" in dados_processados.columns:
    le_tipo = LabelEncoder()
    dados_processados["tipo_encoded"] = le_tipo.fit_transform(dados_processados["tipo"].fillna("Unknown"))
    features_cols.append("tipo_encoded")

# Preparar matriz X
X = dados_processados[features_cols].copy()

# Tratar valores ausentes com estratégia avançada
X = X.fillna(X.median())

st.success(f"✅ {X.shape[1]} features preparadas inicialmente")

# Seleção automática de features
if usar_selecao and n_features:
    st.subheader("🎯 Seleção Automática de Features")
    
    # Preparar y temporário para seleção
    if all(col in dados_processados.columns for col in COLS_FALHA):
        y_temp = dados_processados[COLS_FALHA].max(axis=1)  # Qualquer falha
    else:
        y_temp = dados_processados.get(COL_ALVO_BINARIA, pd.Series([0]*len(dados_processados)))
    
    # Aplicar SelectKBest
    selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
    X_selected = selector.fit_transform(X, y_temp)
    
    # Obter nomes das features selecionadas
    selected_features = [features_cols[i] for i in selector.get_support(indices=True)]
    X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    features_cols = selected_features
    
    st.success(f"✅ {len(selected_features)} features selecionadas automaticamente")
    
    # Mostrar importância das features selecionadas
    scores = selector.scores_[selector.get_support()]
    df_scores = pd.DataFrame({
        'Feature': selected_features,
        'Score': scores
    }).sort_values('Score', ascending=False)
    
    fig_scores = px.bar(
        df_scores.head(15), 
        x='Score', 
        y='Feature',
        orientation='h',
        title="Top Features Selecionadas (F-Score)"
    )
    st.plotly_chart(fig_scores, use_container_width=True)

# Normalização
scaler = None
if normalizar:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# Preparar targets baseado no tipo de modelagem
if tipo_modelagem == "Multirrótulo":
    if not all(col in dados_processados.columns for col in COLS_FALHA):
        st.error("❌ Dados não contêm colunas necessárias para modelagem multirrótulo.")
        st.stop()
    y = dados_processados[COLS_FALHA].values
    target_names = COLS_FALHA
    
elif tipo_modelagem == "Binária":
    if COL_ALVO_BINARIA not in dados_processados.columns:
        st.error("❌ Coluna 'falha_maquina' não encontrada para modelagem binária.")
        st.stop()
    y = dados_processados[COL_ALVO_BINARIA].values
    target_names = ["Sem Falha", "Com Falha"]
    
else:  # Multiclasse
    if not all(col in dados_processados.columns for col in COLS_FALHA):
        st.error("❌ Dados não contêm colunas necessárias para modelagem multiclasse.")
        st.stop()
    
    # Criar target multiclasse
    def get_primary_failure(row):
        failures = [col for col in COLS_FALHA if row[col] == 1]
        if len(failures) == 0:
            return "sem_falha"
        elif len(failures) == 1:
            return failures[0]
        else:
            return "multiplas_falhas"
    
    y_multiclass = dados_processados[COLS_FALHA].apply(get_primary_failure, axis=1)
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_multiclass)
    target_names = le_target.classes_

st.write(f"**Features finais utilizadas:** {len(features_cols)} features")

# =============================================================================
# SEÇÃO 5: MODELAGEM AVANÇADA COM MÚLTIPLOS ALGORITMOS
# =============================================================================
st.header("🤖 5. MODELAGEM AVANÇADA")

# Dividir dados
if tipo_modelagem == "Multirrótulo":
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=percentual_teste/100, random_state=semente
    )
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=percentual_teste/100, random_state=semente, 
        stratify=y if len(np.unique(y)) > 1 else None
    )

st.write(f"📊 **Divisão dos dados:** {len(X_train)} treino / {len(X_val)} validação")

# Treinar modelos
resultados_modelos = {}

with st.spinner("🔄 Treinando modelos avançados..."):
    
    if algoritmo == "Ensemble (Todos)":
        st.subheader("🎯 Ensemble de Modelos")
        modelo = criar_ensemble(X_train, y_train, tipo_modelagem)
        
        if tipo_modelagem == "Multirrótulo":
            # Para multirrótulo, fazer predições com cada modelo do ensemble
            predicoes_ensemble = {}
            for name, model in modelo.items():
                pred = model.predict(X_val)
                predicoes_ensemble[name] = pred
                
                # Calcular métricas médias
                accuracies = [accuracy_score(y_val[:, i], pred[:, i]) for i in range(len(COLS_FALHA))]
                resultados_modelos[name] = np.mean(accuracies)
            
            # Predição final por votação majoritária
            y_pred = np.round(np.mean([pred for pred in predicoes_ensemble.values()], axis=0)).astype(int)
            
        else:
            y_pred = modelo.predict(X_val)
            resultados_modelos["Ensemble"] = accuracy_score(y_val, y_pred)
    
    else:
        # Treinar modelo individual
        modelos_disponiveis = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=semente),
            "XGBoost": xgb.XGBClassifier(n_estimators=200, random_state=semente, eval_metric='logloss'),
            "LightGBM": lgb.LGBMClassifier(n_estimators=200, random_state=semente, verbose=-1),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=semente)
        }
        
        modelo_base = modelos_disponiveis[algoritmo]
        
        # Otimização de hiperparâmetros
        if otimizar_hiper:
            st.write(f"🔍 Otimizando hiperparâmetros com {tipo_busca}...")
            if tipo_modelagem != "Multirrótulo":
                modelo_base = otimizar_hiperparametros(modelo_base, X_train, y_train, algoritmo, tipo_busca)
        
        # Aplicar MultiOutputClassifier se necessário
        if tipo_modelagem == "Multirrótulo":
            modelo = MultiOutputClassifier(modelo_base)
        else:
            modelo = modelo_base
        
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_val)
        
        if tipo_modelagem == "Multirrótulo":
            accuracies = [accuracy_score(y_val[:, i], y_pred[:, i]) for i in range(len(COLS_FALHA))]
            resultados_modelos[algoritmo] = np.mean(accuracies)
        else:
            resultados_modelos[algoritmo] = accuracy_score(y_val, y_pred)

st.success("✅ Modelos treinados com sucesso!")

# Mostrar comparação de modelos se ensemble
if len(resultados_modelos) > 1:
    st.subheader("📊 Comparação de Modelos")
    df_resultados = pd.DataFrame(list(resultados_modelos.items()), columns=['Modelo', 'Accuracy'])
    df_resultados = df_resultados.sort_values('Accuracy', ascending=False)
    
    fig_comp = px.bar(
        df_resultados, 
        x='Accuracy', 
        y='Modelo',
        orientation='h',
        title="Comparação de Performance dos Modelos"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# =============================================================================
# SEÇÃO 6: AVALIAÇÃO AVANÇADA
# =============================================================================
st.header("📊 6. AVALIAÇÃO AVANÇADA DOS RESULTADOS")

if tipo_modelagem == "Multirrótulo":
    # Métricas detalhadas por classe
    st.subheader("📈 Métricas Detalhadas por Tipo de Falha")
    
    metricas_detalhadas = {}
    cols_display = st.columns(len(COLS_FALHA))
    
    for i, col_falha in enumerate(COLS_FALHA):
        accuracy = accuracy_score(y_val[:, i], y_pred[:, i])
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        
        # Calcular AUC se possível
        try:
            if hasattr(modelo, 'predict_proba') or (hasattr(modelo, 'estimators_') and hasattr(modelo.estimators_[i], 'predict_proba')):
                if hasattr(modelo, 'estimators_'):
                    y_proba = modelo.estimators_[i].predict_proba(X_val)[:, 1]
                else:
                    y_proba = modelo.predict_proba(X_val)[i][:, 1]
                auc = roc_auc_score(y_val[:, i], y_proba)
            else:
                auc = 0.5
        except:
            auc = 0.5
        
        metricas_detalhadas[col_falha] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc
        }
        
        with cols_display[i]:
            st.metric(f"🎯 {col_falha.upper()}", f"{accuracy:.3f}")
            st.write(f"Precision: {precision:.3f}")
            st.write(f"Recall: {recall:.3f}")
            st.write(f"F1: {f1:.3f}")
            st.write(f"AUC: {auc:.3f}")
    
    # Métricas médias
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        acc_media = np.mean([metricas_detalhadas[col]['Accuracy'] for col in COLS_FALHA])
        st.metric("🏆 Accuracy Média", f"{acc_media:.4f}")
    with col2:
        prec_media = np.mean([metricas_detalhadas[col]['Precision'] for col in COLS_FALHA])
        st.metric("🎯 Precision Média", f"{prec_media:.4f}")
    with col3:
        rec_media = np.mean([metricas_detalhadas[col]['Recall'] for col in COLS_FALHA])
        st.metric("📊 Recall Médio", f"{rec_media:.4f}")
    with col4:
        f1_media = np.mean([metricas_detalhadas[col]['F1-Score'] for col in COLS_FALHA])
        st.metric("⚖️ F1 Médio", f"{f1_media:.4f}")

else:
    # Modelo binário ou multiclasse
    accuracy = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average='weighted', zero_division=0
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", f"{accuracy:.4f}")
    with col2:
        st.metric("🔍 Precision", f"{precision:.4f}")
    with col3:
        st.metric("📊 Recall", f"{recall:.4f}")
    with col4:
        st.metric("⚖️ F1-Score", f"{f1:.4f}")
    
    # Relatório de classificação
    st.subheader("📋 Relatório Detalhado de Classificação")
    
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.round(4), use_container_width=True)
    
    # Matriz de confusão
    st.subheader("🔍 Matriz de Confusão")
    
    cm = confusion_matrix(y_val, y_pred)
    fig_cm = px.imshow(
        cm, 
        text_auto=True, 
        aspect="auto", 
        title="Matriz de Confusão",
        color_continuous_scale="Blues"
    )
    fig_cm.update_xaxes(title="Predito")
    fig_cm.update_yaxes(title="Real")
    st.plotly_chart(fig_cm, use_container_width=True)

# Feature Importance Avançada
st.subheader("🎯 Análise de Importância das Features")

if algoritmo == "Ensemble (Todos)" and tipo_modelagem != "Multirrótulo":
    # Para ensemble não-multirrótulo
    if hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
    else:
        # Média das importâncias dos estimadores
        importances = np.mean([est.feature_importances_ for name, est in modelo.named_estimators_.items() 
                              if hasattr(est, 'feature_importances_')], axis=0)
        
elif algoritmo == "Ensemble (Todos)" and tipo_modelagem == "Multirrótulo":
    # Para ensemble multirrótulo, usar Random Forest como referência
    if 'rf' in modelo and hasattr(modelo['rf'], 'estimators_'):
        importances = np.mean([est.feature_importances_ for est in modelo['rf'].estimators_], axis=0)
    else:
        importances = np.ones(len(features_cols)) / len(features_cols)  # Uniforme se não disponível
        
elif hasattr(modelo, 'feature_importances_'):
    importances = modelo.feature_importances_
elif tipo_modelagem == "Multirrótulo" and hasattr(modelo, 'estimators_'):
    # Para MultiOutputClassifier
    importances = np.mean([est.feature_importances_ for est in modelo.estimators_ 
                          if hasattr(est, 'feature_importances_')], axis=0)
else:
    importances = np.ones(len(features_cols)) / len(features_cols)  # Uniforme se não disponível

feature_importance_df = pd.DataFrame({
    'Feature': features_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Mostrar top features
col1, col2 = st.columns(2)

with col1:
    fig_importance = px.bar(
        feature_importance_df.head(15), 
        x='Importance', 
        y='Feature',
        orientation='h',
        title="Top 15 Features Mais Importantes"
    )
    fig_importance.update_layout(height=500)
    st.plotly_chart(fig_importance, use_container_width=True)

with col2:
    # Análise das features criadas vs originais
    feature_importance_df['Tipo'] = feature_importance_df['Feature'].apply(
        lambda x: 'Engenharia' if any(keyword in x for keyword in 
                                    ['interacao', 'media_movel', 'desvio_movel', 'tendencia', 
                                     'potencia', 'eficiencia', 'dif_', 'razao_']) 
                               else 'Original'
    )
    
    tipo_importancia = feature_importance_df.groupby('Tipo')['Importance'].sum()
    
    fig_pie = px.pie(
        values=tipo_importancia.values,
        names=tipo_importancia.index,
        title="Contribuição: Features Originais vs Criadas"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# =============================================================================
# SEÇÃO 7: PREDIÇÕES NO CONJUNTO DE TESTE
# =============================================================================
if dados_teste is not None:
    st.header("🎯 7. PREDIÇÕES NO CONJUNTO DE TESTE")
    
    # Processar dados de teste
    dados_teste_proc = criar_features_avancadas(dados_teste)
    
    if "tipo" in dados_teste_proc.columns and 'le_tipo' in locals():
        dados_teste_proc["tipo_encoded"] = le_tipo.transform(
            dados_teste_proc["tipo"].fillna("Unknown")
        )
    
    # Preparar X_test com as mesmas features
    X_test = dados_teste_proc[features_cols].copy()
    X_test = X_test.fillna(X_test.median())
    
    if scaler is not None:
        X_test = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns, 
            index=X_test.index
        )
    
    # Gerar predições
    if tipo_modelagem == "Multirrótulo":
        if algoritmo == "Ensemble (Todos)":
            # Média das predições do ensemble
            predicoes_teste = {}
            for name, model in modelo.items():
                predicoes_teste[name] = model.predict(X_test)
            
            pred_test = np.round(np.mean([pred for pred in predicoes_teste.values()], axis=0)).astype(int)
        else:
            pred_test = modelo.predict(X_test)
        
        # Criar DataFrame de saída
        df_predicoes = pd.DataFrame(
            pred_test, 
            columns=[col.upper() for col in COLS_FALHA]
        ).astype(int)
        
    else:
        pred_test = modelo.predict(X_test)
        
        # Para binário/multiclasse, criar formato multirrótulo
        df_predicoes = pd.DataFrame(
            0, 
            index=range(len(pred_test)), 
            columns=[col.upper() for col in COLS_FALHA]
        )
        
        if tipo_modelagem == "Binária":
            # Atribuir falha mais comum quando há predição positiva
            falha_mais_comum = dados_processados[COLS_FALHA].sum().idxmax().upper()
            df_predicoes.loc[pred_test == 1, falha_mais_comum] = 1
            
        else:  # Multiclasse
            for i, pred in enumerate(pred_test):
                classe_nome = target_names[pred]
                if classe_nome in [col.lower() for col in COLS_FALHA]:
                    col_idx = [col.lower() for col in COLS_FALHA].index(classe_nome)
                    df_predicoes.iloc[i, col_idx] = 1
    
    # Estatísticas das predições
    st.subheader("📊 Estatísticas das Predições")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Contagem por tipo de falha:**")
        contagem_pred = df_predicoes.sum().to_frame("Quantidade")
        contagem_pred["Percentual"] = (contagem_pred["Quantidade"] / len(df_predicoes) * 100).round(2)
        st.dataframe(contagem_pred, use_container_width=True)
    
    with col2:
        # Gráfico de barras das predições
        fig_pred = px.bar(
            x=df_predicoes.sum().values,
            y=df_predicoes.sum().index,
            orientation='h',
            title="Distribuição das Predições por Tipo de Falha"
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    
    # Análise comparativa (se tivermos dados históricos)
    st.subheader("📈 Análise Comparativa")
    col1, col2 = st.columns(2)
    
    with col1:
        # Comparar distribuição treino vs predições
        dist_treino = dados_processados[COLS_FALHA].sum()
        dist_pred = df_predicoes.sum()
        
        df_comparacao = pd.DataFrame({
            'Treino': dist_treino.values,
            'Predições': dist_pred.values,
            'Falha': [col.upper() for col in COLS_FALHA]
        })
        
        fig_comp = px.bar(
            df_comparacao.melt(id_vars='Falha', var_name='Dataset', value_name='Quantidade'),
            x='Falha', y='Quantidade', color='Dataset', barmode='group',
            title="Comparação: Distribuição Treino vs Predições"
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    
    with col2:
        # Métricas de confiança (se modelo suporta probabilidades)
        if hasattr(modelo, 'predict_proba') or (algoritmo == "Ensemble (Todos)" and tipo_modelagem != "Multirrótulo"):
            st.write("**Métricas de Confiança:**")
            
            try:
                if tipo_modelagem == "Multirrótulo":
                    # Para multirrótulo, calcular confiança média
                    confidencias = []
                    if algoritmo == "Ensemble (Todos)":
                        for name, model in modelo.items():
                            if hasattr(model, 'estimators_'):
                                conf = np.mean([est.predict_proba(X_test)[:, 1] if hasattr(est, 'predict_proba') else 0.5
                                              for est in model.estimators_], axis=0)
                                confidencias.append(conf)
                        confianca_media = np.mean(confidencias) if confidencias else 0.5
                    else:
                        confianca_media = 0.5
                else:
                    if hasattr(modelo, 'predict_proba'):
                        probas = modelo.predict_proba(X_test)
                        confianca_media = np.mean(np.max(probas, axis=1))
                    else:
                        confianca_media = 0.5
                
                st.metric("🎯 Confiança Média", f"{confianca_media:.3f}")
                
                # Distribuição de confiança
                if tipo_modelagem != "Multirrótulo" and hasattr(modelo, 'predict_proba'):
                    conf_dist = np.max(modelo.predict_proba(X_test), axis=1)
                    fig_conf = px.histogram(x=conf_dist, title="Distribuição da Confiança")
                    st.plotly_chart(fig_conf, use_container_width=True)
                    
            except Exception as e:
                st.write("Confiança não disponível para este modelo")
    
    # Download das predições
    csv_predicoes = df_predicoes.to_csv(index=False)
    st.download_button(
        "📥 Baixar Predições (CSV para API)",
        csv_predicoes,
        "bootcamp_predictions.csv",
        "text/csv"
    )
    
    st.success("✅ Predições geradas com sucesso!")

# =============================================================================
# SEÇÃO 8: ANÁLISE DE PERFORMANCE E INSIGHTS
# =============================================================================
st.header("📊 8. ANÁLISE DE PERFORMANCE E INSIGHTS")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Insights do Modelo")
    
    insights = []
    
    # Insight sobre features mais importantes
    top_feature = feature_importance_df.iloc[0]
    insights.append(f"🎯 **Feature mais importante:** {top_feature['Feature']} ({top_feature['Importance']:.3f})")
    
    # Insight sobre engenharia de features
    eng_features = feature_importance_df[feature_importance_df['Tipo'] == 'Engenharia']
    if not eng_features.empty:
        contrib_eng = feature_importance_df.groupby('Tipo')['Importance'].sum().get('Engenharia', 0)
        insights.append(f"🔧 **Contribuição das features criadas:** {contrib_eng:.1%}")
    
    # Insight sobre desbalanceamento
    if 'falhas_count' in locals():
        ratio = falhas_count.max() / falhas_count.min()
        insights.append(f"⚖️ **Desbalanceamento:** {ratio:.1f}:1 (classe majoritária vs minoritária)")
    
    # Insight sobre performance
    if tipo_modelagem == "Multirrótulo":
        melhor_classe = max(metricas_detalhadas.keys(), 
                           key=lambda x: metricas_detalhadas[x]['F1-Score'])
        pior_classe = min(metricas_detalhadas.keys(), 
                         key=lambda x: metricas_detalhadas[x]['F1-Score'])
        insights.append(f"📈 **Melhor predição:** {melhor_classe.upper()} (F1: {metricas_detalhadas[melhor_classe]['F1-Score']:.3f})")
        insights.append(f"📉 **Maior desafio:** {pior_classe.upper()} (F1: {metricas_detalhadas[pior_classe]['F1-Score']:.3f})")
    
    for insight in insights:
        st.markdown(insight)

with col2:
    st.subheader("💡 Recomendações")
    
    recomendacoes = []
    
    # Recomendação baseada na performance
    if tipo_modelagem == "Multirrótulo":
        f1_scores = [metricas_detalhadas[col]['F1-Score'] for col in COLS_FALHA]
        f1_medio = np.mean(f1_scores)
        if f1_medio < 0.7:
            recomendacoes.append("📊 Considere técnicas de balanceamento (SMOTE, undersampling)")
        if min(f1_scores) < 0.5:
            recomendacoes.append("🎯 Foque na coleta de mais dados para classes minoritárias")
    
    # Recomendação sobre features
    if len(eng_features) > 0:
        recomendacoes.append("🔧 Features de engenharia mostraram-se valiosas - continue explorando")
    
    # Recomendação sobre algoritmos
    if algoritmo != "Ensemble (Todos)":
        recomendacoes.append("🤖 Teste ensemble de modelos para melhor performance")
    
    recomendacoes.extend([
        "📈 Implemente monitoramento contínuo da performance",
        "🔄 Considere retreinamento periódico com novos dados",
        "⚡ Desenvolva sistema de alertas baseado nas predições",
        "🏭 Integre com sistema de gestão de manutenção existente"
    ])
    
    for rec in recomendacoes:
        st.markdown(f"- {rec}")

# =============================================================================
# SEÇÃO 9: CONCLUSÕES E PRÓXIMOS PASSOS
# =============================================================================
st.header("📋 9. CONCLUSÕES E PRÓXIMOS PASSOS")

with st.expander("🎯 Principais Conclusões", expanded=True):
    st.markdown(f"""
    ### ✅ **Resultados Alcançados:**
    
    1. **Sistema Completo de Manutenção Preditiva:**
       - Análise exploratória avançada com {dados_processados.shape[1]} features
       - Engenharia de features criou {dados_processados.shape[1] - dados_treino.shape[1]} novas variáveis
       - Implementação de múltiplos algoritmos: Random Forest, XGBoost, LightGBM, Neural Networks
       - Ensemble de modelos para máxima performance
    
    2. **Performance do Modelo:**
       - Algoritmo utilizado: **{algoritmo}**
       - Abordagem: **{tipo_modelagem}**
       - Otimização de hiperparâmetros: **{'Sim' if otimizar_hiper else 'Não'}**
       - Features selecionadas: **{len(features_cols)}**
    
    3. **Inovações Implementadas:**
       - 🔧 Engenharia avançada de features (interações, séries temporais simuladas)
       - 🎯 Seleção automática de features
       - ⚙️ Otimização de hiperparâmetros
       - 🤖 Ensemble de múltiplos algoritmos
       - 📊 Análise comparativa de performance
    
    4. **Impacto no Negócio:**
       - Sistema pronto para detecção precoce de falhas
       - Redução esperada de custos de manutenção
       - Melhoria na disponibilidade dos equipamentos
       - Interface intuitiva para operadores
    """)

with st.expander("🚀 Roadmap de Melhorias Futuras", expanded=False):
    st.markdown("""
    ### 📈 **Próximas Fases do Projeto:**
    
    **Fase 1 - Produção (Próximos 30 dias):**
    - ⚡ API RESTful com FastAPI para integração
    - 🐳 Containerização com Docker
    - 📊 Dashboard em tempo real para monitoramento
    - 🔒 Sistema de autenticação e logs
    
    **Fase 2 - Inteligência Avançada (60-90 dias):**
    - 🧠 Deep Learning com redes neurais recorrentes (LSTM)
    - 📈 Análise de séries temporais reais
    - 🎯 Detecção de anomalias não supervisionada
    - 🔄 Auto-ML para otimização contínua
    
    **Fase 3 - Integração Empresarial (90-120 dias):**
    - 🏭 Integração com sistemas ERP/MES
    - 📱 App mobile para técnicos de manutenção
    - 🤖 Chatbot para consultas sobre equipamentos
    - 📊 Dashboards executivos com KPIs de negócio
    
    **Fase 4 - Escala e Otimização (120+ dias):**
    - ☁️ Deploy em nuvem com auto-scaling
    - 🔄 Pipeline de CI/CD completo
    - 📈 MLOps com monitoramento de drift
    - 🌐 Multi-tenancy para diferentes plantas industriais
    """)

# =============================================================================
# MÉTRICAS FINAIS DE SISTEMA
# =============================================================================
st.header("📊 Dashboard Executivo")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Amostras Processadas", f"{dados_processados.shape[0]:,}")
with col2:
    st.metric("🔧 Features Totais", dados_processados.shape[1])
with col3:
    if tipo_modelagem == "Multirrótulo":
        perf_media = np.mean([metricas_detalhadas[col]['F1-Score'] for col in COLS_FALHA])
        st.metric("🎯 Performance Média", f"{perf_media:.3f}")
    else:
        st.metric("🎯 Accuracy Final", f"{accuracy:.3f}")
with col4:
    if dados_teste is not None:
        falhas_previstas = df_predicoes.sum().sum()
        st.metric("⚠️ Falhas Previstas", int(falhas_previstas))
    else:
        st.metric("💾 Modelo Treinado", "✅")

# =============================================================================
# RODAPÉ PROFISSIONAL
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>🔧 Sistema Inteligente de Manutenção Preditiva</h4>
    <p><strong>Bootcamp de Ciência de Dados e IA - Projeto Final</strong></p>
    <p>Desenvolvido com ❤️ usando Python, Scikit-learn, XGBoost, LightGBM e Streamlit</p>
    <p><em>Implementação completa com algoritmos avançados, engenharia de features e ensemble de modelos</em></p>
</div>
""", unsafe_allow_html=True)
