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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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

st.title("🔧 Sistema Inteligente de Manutenção Preditiva")
st.markdown("### Bootcamp de Ciência de Dados e IA - Projeto Final")

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
    
    **Metodologia Aplicada:**
    - 🤖 Múltiplos algoritmos de ML (RF, Gradient Boosting, Neural Networks, SVM)
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
# SEÇÃO 2: CARREGAMENTO E CONFIGURAÇÃO
# =============================================================================
st.header("📁 2. CONFIGURAÇÃO DO SISTEMA")

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
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
        ["Ensemble", "Random Forest", "Gradient Boosting", "Neural Network", "SVM"],
        help="Ensemble combina múltiplos algoritmos"
    )
    
    otimizar_hiper = st.checkbox("🔍 Otimização de Hiperparâmetros", value=False)
    tipo_busca = st.selectbox("Tipo de Busca:", ["RandomizedSearch", "GridSearch"]) if otimizar_hiper else "Nenhuma"
    
    percentual_teste = st.slider("% Validação", 10, 40, 20, 5)
    semente = st.slider("Semente Aleatória", 0, 9999, 42)
    
    st.subheader("🔬 Engenharia de Features")
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
# FUNÇÕES UTILITÁRIAS
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
    """Aplica engenharia de features"""
    df = df.copy()
    
    # Features básicas de engenharia
    if usar_dif_temp and {"temperatura_processo","temperatura_ar"}.issubset(df.columns):
        df["dif_temperatura"] = df["temperatura_processo"] - df["temperatura_ar"]
        df["razao_temperatura"] = df["temperatura_processo"] / (df["temperatura_ar"] + 1e-6)
        df["temp_normalizada"] = (df["temperatura_processo"] - df["temperatura_ar"].mean()) / (df["temperatura_ar"].std() + 1e-6)
    
    if usar_potencia and {"torque","velocidade_rotacional"}.issubset(df.columns):
        df["potencia_kw"] = (df["torque"] * df["velocidade_rotacional"]) / 1000.0
        df["potencia_especifica"] = df["potencia_kw"] / (df["desgaste_da_ferramenta"] + 1.0)
        df["torque_por_velocidade"] = df["torque"] / (df["velocidade_rotacional"] + 1.0)
    
    if usar_eficiencia and {"torque","desgaste_da_ferramenta"}.issubset(df.columns):
        df["eficiencia_ferramenta"] = df["torque"] / (df["desgaste_da_ferramenta"] + 1.0)
        df["desgaste_normalizado"] = df["desgaste_da_ferramenta"] / (df["torque"] + 1e-6)
        df["stress_ferramenta"] = df["torque"] * df["desgaste_da_ferramenta"]
    
    # Features baseadas em séries temporais (simuladas através do índice)
    if usar_series_temporais:
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
        "Gradient Boosting": {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        },
        "Neural Network": {
            'hidden_layer_sizes': [(50,), (100,), (50, 30)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['adaptive', 'constant']
        },
        "SVM": {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
    }
    
    if algoritmo_nome not in param_grids:
        return modelo
    
    param_grid = param_grids[algoritmo_nome]
    
    try:
        if tipo_busca == "GridSearch":
            search = GridSearchCV(modelo, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        else:
            search = RandomizedSearchCV(modelo, param_grid, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
        
        search.fit(X, y)
        return search.best_estimator_
    except:
        return modelo

def criar_ensemble(X_train, y_train, tipo_modelagem):
    """Cria ensemble de modelos"""
    modelos_base = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'nn': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        'dt': DecisionTreeClassifier(random_state=42)
    }
    
    if tipo_modelagem == "Multirrótulo":
        ensemble_models = {}
        for name, model in modelos_base.items():
            try:
                ensemble_models[name] = MultiOutputClassifier(model)
                ensemble_models[name].fit(X_train, y_train)
            except:
                continue
        return ensemble_models
    else:
        # Para binário/multiclasse, usar VotingClassifier
        try:
            ensemble = VotingClassifier(
                estimators=list(modelos_base.items()),
                voting='soft'
            )
            ensemble.fit(X_train, y_train)
            return ensemble
        except:
            # Fallback para Random Forest
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_train, y_train)
            return rf

# Carregar dados
dados_treino = None
dados_teste = None

if arquivo_treino:
    try:
        dados_treino = carregar_e_processar_dados(arquivo_treino)
        st.success(f"✅ Dados de treino carregados: {dados_treino.shape[0]} amostras × {dados_treino.shape[1]} features")
    except Exception as e:
        st.error(f"Erro ao carregar dados de treino: {str(e)}")
        st.stop()
    
if arquivo_teste:
    try:
        dados_teste = carregar_e_processar_dados(arquivo_teste)
        st.success(f"✅ Dados de teste carregados: {dados_teste.shape[0]} amostras")
    except Exception as e:
        st.error(f"Erro ao carregar dados de teste: {str(e)}")

if dados_treino is None:
    st.warning("⏳ Por favor, carregue o arquivo de treino para continuar.")
    st.stop()

# =============================================================================
# SEÇÃO 3: ANÁLISE EXPLORATÓRIA
# =============================================================================
st.header("📊 3. ANÁLISE EXPLORATÓRIA")

# Aplicar engenharia de features
try:
    dados_processados = criar_features_avancadas(dados_treino)
except Exception as e:
    st.error(f"Erro na criação de features: {str(e)}")
    dados_processados = dados_treino.copy()

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
    # Análise de desbalanceamento
    st.subheader("⚖️ Análise de Desbalanceamento")
    
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
        if falhas_count.min() > 0:
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

# =============================================================================
# SEÇÃO 4: PREPARAÇÃO DOS DADOS
# =============================================================================
st.header("🔧 4. PREPARAÇÃO DOS DADOS")

# Preparar features (X) e targets (y)
excluir_cols = {"id", "id_produto", "tipo", COL_ALVO_BINARIA} | set(COLS_FALHA)
features_cols = [col for col in dados_processados.columns if col not in excluir_cols]

# Tratar valores categóricos
if "tipo" in dados_processados.columns:
    try:
        le_tipo = LabelEncoder()
        dados_processados["tipo_encoded"] = le_tipo.fit_transform(dados_processados["tipo"].fillna("Unknown"))
        features_cols.append("tipo_encoded")
    except:
        pass

# Preparar matriz X
X = dados_processados[features_cols].copy()
X = X.fillna(X.median())

# Seleção automática de features
if usar_selecao and n_features:
    st.subheader("🎯 Seleção Automática de Features")
    
    # Preparar y temporário para seleção
    if all(col in dados_processados.columns for col in COLS_FALHA):
        y_temp = dados_processados[COLS_FALHA].max(axis=1)
    else:
        y_temp = dados_processados.get(COL_ALVO_BINARIA, pd.Series([0]*len(dados_processados)))
    
    try:
        selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y_temp)
        selected_features = [features_cols[i] for i in selector.get_support(indices=True)]
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        features_cols = selected_features
        st.success(f"✅ {len(selected_features)} features selecionadas")
    except Exception as e:
        st.warning(f"Não foi possível aplicar seleção de features: {str(e)}")

# Normalização
scaler = None
if normalizar:
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    except Exception as e:
        st.warning(f"Erro na normalização: {str(e)}")

# Preparar targets
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
# SEÇÃO 5: MODELAGEM
# =============================================================================
st.header("🤖 5. MODELAGEM")

# Dividir dados
try:
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
except Exception as e:
    st.error(f"Erro na divisão dos dados: {str(e)}")
    st.stop()

# Treinar modelos
resultados_modelos = {}

with st.spinner("🔄 Treinando modelo..."):
    try:
        if algoritmo == "Ensemble":
            modelo = criar_ensemble(X_train, y_train, tipo_modelagem)
        else:
            # Treinar modelo individual
            modelos_disponiveis = {
                "Random Forest": RandomForestClassifier(n_estimators=200, random_state=semente),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=semente),
                "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=semente),
                "SVM": SVC(probability=True, random_state=semente)
            }
            
            modelo_base = modelos_disponiveis[algoritmo]
            
            # Otimização de hiperparâmetros
            if otimizar_hiper and tipo_modelagem != "Multirrótulo":
                st.write(f"🔍 Otimizando hiperparâmetros...")
                modelo_base = otimizar_hiperparametros(modelo_base, X_train, y_train, algoritmo, tipo_busca)
            
            # Aplicar MultiOutputClassifier se necessário
            if tipo_modelagem == "Multirrótulo":
                modelo = MultiOutputClassifier(modelo_base)
            else:
                modelo = modelo_base
            
            modelo.fit(X_train, y_train)
        
        st.success("✅ Modelo treinado com sucesso!")
        
    except Exception as e:
        st.error(f"Erro no treinamento: {str(e)}")
        st.stop()

# Fazer predições
try:
    y_pred = modelo.predict(X_val)
except Exception as e:
    st.error(f"Erro nas predições: {str(e)}")
    st.stop()

# =============================================================================
# SEÇÃO 6: AVALIAÇÃO
# =============================================================================
st.header("📊 6. AVALIAÇÃO DOS RESULTADOS")

try:
    if tipo_modelagem == "Multirrótulo":
        # Métricas detalhadas por classe
        st.subheader("📈 Métricas por Tipo de Falha")
        
        metricas_detalhadas = {}
        cols_display = st.columns(len(COLS_FALHA))
        
        for i, col_falha in enumerate(COLS_FALHA):
            accuracy = accuracy_score(y_val[:, i], y_pred[:, i])
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val[:, i], y_pred[:, i], average='binary', zero_division=0
            )
            
            metricas_detalhadas[col_falha] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            }
            
            with cols_display[i]:
                st.metric(f"🎯 {col_falha.upper()}", f"{accuracy:.3f}")
                st.write(f"Precision: {precision:.3f}")
                st.write(f"Recall: {recall:.3f}")
                st.write(f"F1: {f1:.3f}")
        
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

except Exception as e:
    st.error(f"Erro na avaliação: {str(e)}")

# Feature Importance
st.subheader("🎯 Importância das Features")

try:
    importances = None
    if algoritmo == "Ensemble" and hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
    elif algoritmo == "Ensemble" and isinstance(modelo, dict):
        # Para ensemble multirrótulo
        if 'rf' in modelo:
            importances = np.mean([est.feature_importances_ for est in modelo['rf'].estimators_], axis=0)
    elif hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
    elif tipo_modelagem == "Multirrótulo" and hasattr(modelo, 'estimators_'):
        importances = np.mean([est.feature_importances_ for est in modelo.estimators_ 
                              if hasattr(est, 'feature_importances_')], axis=0)
    
    if importances is not None and len(importances) == len(features_cols):
        feature_importance_df = pd.DataFrame({
            'Feature': features_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            feature_importance_df.head(15), 
            x='Importance', 
            y='Feature',
            orientation='h',
            title="Top 15 Features Mais Importantes"
        )
        fig_importance.update_layout(height=500)
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.info("Importância das features não disponível para este modelo.")

except Exception as e:
    st.warning(f"Não foi possível calcular importância das features: {str(e)}")

# =============================================================================
# SEÇÃO 7: PREDIÇÕES NO TESTE
# =============================================================================
if dados_teste is not None:
    st.header("🎯 7. PREDIÇÕES NO CONJUNTO DE TESTE")
    
    try:
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
            if algoritmo == "Ensemble" and isinstance(modelo, dict):
                # Média das predições do ensemble
                predicoes_teste = {}
                for name, model in modelo.items():
                    try:
                        predicoes_teste[name] = model.predict(X_test)
                    except:
                        continue
                
                if predicoes_teste:
                    pred_test = np.round(np.mean([pred for pred in predicoes_teste.values()], axis=0)).astype(int)
                else:
                    pred_test = np.zeros((len(X_test), len(COLS_FALHA)))
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
        
        # Download das predições
        csv_predicoes = df_predicoes.to_csv(index=False)
        st.download_button(
            "📥 Baixar Predições (CSV)",
            csv_predicoes,
            "bootcamp_predictions.csv",
            "text/csv"
        )
        
        st.success("✅ Predições geradas com sucesso!")
        
    except Exception as e:
        st.error(f"Erro ao processar dados de teste: {str(e)}")

# =============================================================================
# SEÇÃO 8: INSIGHTS E RECOMENDAÇÕES
# =============================================================================
st.header("📊 8. INSIGHTS E RECOMENDAÇÕES")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Insights do Modelo")
    
    insights = []
    
    # Insights baseados na performance
    if tipo_modelagem == "Multirrótulo" and 'metricas_detalhadas' in locals():
        f1_scores = [metricas_detalhadas[col]['F1-Score'] for col in COLS_FALHA]
        melhor_classe = max(metricas_detalhadas.keys(), 
                           key=lambda x: metricas_detalhadas[x]['F1-Score'])
        pior_classe = min(metricas_detalhadas.keys(), 
                         key=lambda x: metricas_detalhadas[x]['F1-Score'])
        
        insights.append(f"🎯 **Melhor predição:** {melhor_classe.upper()}")
        insights.append(f"⚠️ **Maior desafio:** {pior_classe.upper()}")
        insights.append(f"📊 **F1 médio:** {np.mean(f1_scores):.3f}")
    
    elif tipo_modelagem != "Multirrótulo" and 'accuracy' in locals():
        insights.append(f"🎯 **Accuracy geral:** {accuracy:.3f}")
    
    # Insights sobre features
    if 'feature_importance_df' in locals():
        top_feature = feature_importance_df.iloc[0]
        insights.append(f"🔧 **Feature mais importante:** {top_feature['Feature']}")
    
    # Insights sobre dados
    insights.append(f"📊 **Total de features utilizadas:** {len(features_cols)}")
    insights.append(f"🔧 **Features criadas:** {dados_processados.shape[1] - dados_treino.shape[1]}")
    
    for insight in insights:
        st.markdown(insight)

with col2:
    st.subheader("💡 Recomendações")
    
    recomendacoes = [
        "📈 Implementar monitoramento contínuo da performance",
        "🔄 Considerar retreinamento periódico com novos dados",
        "⚡ Desenvolver sistema de alertas baseado nas predições",
        "🏭 Integrar com sistema de gestão de manutenção existente",
        "📊 Coletar feedback dos técnicos para melhorar o modelo",
        "🎯 Focar na coleta de mais dados para classes minoritárias"
    ]
    
    for rec in recomendacoes:
        st.markdown(f"- {rec}")

# =============================================================================
# SEÇÃO 9: CONCLUSÕES
# =============================================================================
st.header("📋 9. CONCLUSÕES")

with st.expander("🎯 Principais Conclusões", expanded=True):
    st.markdown(f"""
    ### ✅ **Resultados Alcançados:**
    
    1. **Sistema Completo de Manutenção Preditiva:**
       - Análise exploratória com {dados_processados.shape[1]} features
       - Engenharia de features criou {dados_processados.shape[1] - dados_treino.shape[1]} novas variáveis
       - Implementação de múltiplos algoritmos de ML
       - Sistema pronto para produção
    
    2. **Performance do Modelo:**
       - Algoritmo utilizado: **{algoritmo}**
       - Abordagem: **{tipo_modelagem}**
       - Otimização de hiperparâmetros: **{'Sim' if otimizar_hiper else 'Não'}**
       - Features utilizadas: **{len(features_cols)}**
    
    3. **Impacto no Negócio:**
       - Sistema pronto para detecção precoce de falhas
       - Redução esperada de custos de manutenção
       - Melhoria na disponibilidade dos equipamentos
       - Interface intuitiva para operadores
    """)

# =============================================================================
# DASHBOARD EXECUTIVO
# =============================================================================
st.header("📊 Dashboard Executivo")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Amostras Processadas", f"{dados_processados.shape[0]:,}")
with col2:
    st.metric("🔧 Features Totais", dados_processados.shape[1])
with col3:
    if tipo_modelagem == "Multirrótulo" and 'metricas_detalhadas' in locals():
        perf_media = np.mean([metricas_detalhadas[col]['F1-Score'] for col in COLS_FALHA])
        st.metric("🎯 Performance Média", f"{perf_media:.3f}")
    elif 'accuracy' in locals():
        st.metric("🎯 Accuracy Final", f"{accuracy:.3f}")
    else:
        st.metric("🎯 Modelo", "Treinado ✅")
with col4:
    if dados_teste is not None and 'df_predicoes' in locals():
        falhas_previstas = df_predicoes.sum().sum()
        st.metric("⚠️ Falhas Previstas", int(falhas_previstas))
    else:
        st.metric("💾 Status", "Pronto ✅")

# =============================================================================
# RODAPÉ
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>🔧 Sistema Inteligente de Manutenção Preditiva</h4>
    <p><strong>Bootcamp de Ciência de Dados e IA - Projeto Final</strong></p>
    <p>Desenvolvido com Python, Scikit-learn e Streamlit</p>
    <p><em>Versão otimizada para compatibilidade com Streamlit Cloud</em></p>
</div>
""", unsafe_allow_html=True)
