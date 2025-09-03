import os, io, numpy as np, pandas as pd, streamlit as st, requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------- Configuração da Página ----------------
st.set_page_config(page_title="Sistema de Manutenção Preditiva", page_icon="🔧", layout="wide")
st.title("🔧 Sistema Inteligente de Manutenção Preditiva")
st.markdown("### Bootcamp de Ciência de Dados e IA - Projeto Final")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Upload de arquivos
    arquivo_treino = st.file_uploader("📁 Selecione o Bootcamp_train.csv", type=["csv"])
    arquivo_teste = st.file_uploader("📁 Selecione o Bootcamp_test.csv (opcional)", type=["csv"])
    
    # Configurações do modelo
    st.subheader("🤖 Parâmetros do Modelo")
    tipo_modelagem = st.selectbox("Tipo de Modelagem:", 
                                 ["Multirrótulo (Cada tipo de falha)", 
                                  "Binária (Qualquer falha)", 
                                  "Multiclasse (Tipo principal de falha)"])
    
    percentual_teste = st.slider("% Conjunto de Validação", 10, 40, 20, step=5)
    semente_aleatoria = st.slider("Semente Aleatória", 0, 999, 42)
    n_estimators = st.slider("Número de Árvores", 100, 500, 200, step=50)
    
    # Feature Engineering
    st.subheader("🔧 Engenharia de Features")
    usar_dif_temp = st.checkbox("Diferença de temperatura", value=True)
    usar_potencia = st.checkbox("Potência (torque × velocidade)", value=True)
    usar_eficiencia = st.checkbox("Eficiência da máquina", value=True)
    normalizar_dados = st.checkbox("Normalizar features numéricas", value=False)

# ---------------- Funções Auxiliares ----------------
@st.cache_data
def carregar_dados(arquivo_treino=None, arquivo_teste=None):
    """Carrega os dados de treino e teste"""
    dados_treino = None
    dados_teste = None
    mensagens = []
    
    # Tentar carregar arquivo de treino
    if arquivo_treino is not None:
        try:
            dados_treino = pd.read_csv(arquivo_treino)
            mensagens.append(("success", f"✅ Treino carregado: {dados_treino.shape[0]} amostras, {dados_treino.shape[1]} colunas"))
        except Exception as e:
            mensagens.append(("error", f"❌ Erro ao carregar treino: {e}"))
    
    # Tentar carregar arquivo de teste
    if arquivo_teste is not None:
        try:
            dados_teste = pd.read_csv(arquivo_teste)
            mensagens.append(("success", f"✅ Teste carregado: {dados_teste.shape[0]} amostras, {dados_teste.shape[1]} colunas"))
        except Exception as e:
            mensagens.append(("error", f"❌ Erro ao carregar teste: {e}"))
    
    return dados_treino, dados_teste, mensagens

def criar_features_engenharia(df, incluir_dif_temp=True, incluir_potencia=True, incluir_eficiencia=True):
    """Cria novas features baseadas na engenharia de características"""
    df_eng = df.copy()
    
    if incluir_dif_temp and 'temperatura_ar' in df.columns and 'temperatura_processo' in df.columns:
        df_eng['diferenca_temperatura'] = df['temperatura_processo'] - df['temperatura_ar']
        df_eng['razao_temperatura'] = df['temperatura_processo'] / (df['temperatura_ar'] + 1e-6)
    
    if incluir_potencia and 'torque' in df.columns and 'velocidade_rotacional' in df.columns:
        df_eng['potencia'] = df['torque'] * df['velocidade_rotacional'] / 1000  # kW
    
    if incluir_eficiencia and 'torque' in df.columns and 'desgaste_da_ferramenta' in df.columns:
        df_eng['eficiencia_ferramenta'] = df['torque'] / (df['desgaste_da_ferramenta'] + 1)
        
    return df_eng

def preparar_targets(df):
    """Prepara os diferentes tipos de target baseado no tipo de modelagem"""
    targets = {}
    
    # Colunas de falha
    colunas_falha = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
    
    # Target binário (qualquer falha)
    if all(col in df.columns for col in colunas_falha):
        targets['binario'] = df[colunas_falha].max(axis=1)
        
        # Target multiclasse (tipo principal de falha)
        def tipo_principal_falha(row):
            falhas = [col for col in colunas_falha if row[col] == 1]
            if len(falhas) == 0:
                return 'Sem_Falha'
            elif len(falhas) == 1:
                return falhas[0]
            else:
                return 'Multiplas_Falhas'
        
        targets['multiclasse'] = df[colunas_falha].apply(tipo_principal_falha, axis=1)
        
        # Target multirrótulo (cada tipo de falha)
        targets['multirrotulo'] = df[colunas_falha].values
        targets['colunas_multirrotulo'] = colunas_falha
        
    return targets

def plotar_distribuicao_falhas(df):
    """Plota a distribuição dos tipos de falha"""
    colunas_falha = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
    nomes_falha = {
        'FDF': 'Desgaste Ferramenta',
        'FDC': 'Dissipação Calor', 
        'FP': 'Falha Potência',
        'FTE': 'Tensão Excessiva',
        'FA': 'Falha Aleatória'
    }
    
    if all(col in df.columns for col in colunas_falha):
        contagens = df[colunas_falha].sum().sort_values(ascending=True)
        
        fig = px.bar(
            x=contagens.values, 
            y=[nomes_falha.get(col, col) for col in contagens.index],
            orientation='h',
            title="Distribuição dos Tipos de Falha",
            labels={'x': 'Quantidade de Falhas', 'y': 'Tipo de Falha'},
            color=contagens.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400, showlegend=False)
        return fig
    return None

def plotar_correlacao_features(df):
    """Plota matriz de correlação das features numéricas"""
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    colunas_features = [col for col in colunas_numericas if not col.startswith(('id', 'FD', 'FP', 'FTE', 'FA', 'falha'))]
    
    if len(colunas_features) > 1:
        corr_matrix = df[colunas_features].corr()
        
        fig = px.imshow(
            corr_matrix, 
            title="Matriz de Correlação das Features",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(height=500)
        return fig
    return None

def treinar_modelo_multirrotulo(X_treino, y_treino, X_valid, y_valid, colunas_falha, n_estimators=200):
    """Treina modelo multirrótulo (um classificador para cada tipo de falha)"""
    from sklearn.multioutput import MultiOutputClassifier
    
    modelo_base = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    modelo = MultiOutputClassifier(modelo_base)
    
    modelo.fit(X_treino, y_treino)
    pred_valid = modelo.predict(X_valid)
    
    # Calcular métricas para cada classe
    metricas = {}
    for i, classe in enumerate(colunas_falha):
        acc = accuracy_score(y_valid[:, i], pred_valid[:, i])
        metricas[classe] = acc
    
    return modelo, pred_valid, metricas

# ---------------- Carregamento dos Dados ----------------
dados_treino, dados_teste, mensagens = carregar_dados(arquivo_treino, arquivo_teste)

for tipo_msg, texto_msg in mensagens:
    getattr(st, tipo_msg)(texto_msg)

if dados_treino is None:
    st.warning("⚠️ Por favor, carregue o arquivo Bootcamp_train.csv para continuar")
    st.stop()

# ---------------- Análise Exploratória ----------------
st.header("📊 Análise Exploratória dos Dados")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total de Amostras", f"{dados_treino.shape[0]:,}")
with col2:
    st.metric("Features", dados_treino.shape[1])
with col3:
    if 'falha_maquina' in dados_treino.columns:
        # Converter a coluna para numérico
        dados_treino['falha_maquina'] = pd.to_numeric(
            dados_treino['falha_maquina'], errors='coerce'
        )

        # Calcular a taxa de falha (considerando apenas valores válidos)
        taxa_falha = dados_treino['falha_maquina'].mean(skipna=True)
        st.metric("Taxa de Falha", f"{taxa_falha:.1%}")

with col4:
    tipos_maquina = dados_treino['tipo'].nunique() if 'tipo' in dados_treino.columns else 0
    st.metric("Tipos de Máquina", tipos_maquina)

# Plots exploratórios
col1, col2 = st.columns(2)
with col1:
    fig_falhas = plotar_distribuicao_falhas(dados_treino)
    if fig_falhas:
        st.plotly_chart(fig_falhas, use_container_width=True)

with col2:
    fig_corr = plotar_correlacao_features(dados_treino)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)

# ---------------- Preparação dos Dados ----------------
st.header("🔧 Preparação dos Dados")

# Aplicar engenharia de features
dados_treino_eng = criar_features_engenharia(
    dados_treino, 
    incluir_dif_temp=usar_dif_temp,
    incluir_potencia=usar_potencia, 
    incluir_eficiencia=usar_eficiencia
)

# Preparar features
colunas_excluir = {'id', 'id_produto', 'tipo', 'falha_maquina', 'FDF', 'FDC', 'FP', 'FTE', 'FA'}
features_numericas = [col for col in dados_treino_eng.columns if col not in colunas_excluir]

# Encoding da variável tipo se existir
le_tipo = None
if 'tipo' in dados_treino_eng.columns:
    le_tipo = LabelEncoder()
    dados_treino_eng['tipo_encoded'] = le_tipo.fit_transform(dados_treino_eng['tipo'])
    features_numericas.append('tipo_encoded')

X = dados_treino_eng[features_numericas]

# Preparar targets
targets = preparar_targets(dados_treino_eng)

# Normalização se solicitada
scaler = None
if normalizar_dados:
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

st.success(f"✅ Dados preparados: {len(features_numericas)} features, {X.shape[0]} amostras")

# ---------------- Treinamento do Modelo ----------------
st.header("🤖 Treinamento do Modelo")

# Selecionar tipo de target baseado na modelagem escolhida
if tipo_modelagem.startswith("Multirrótulo"):
    if 'multirrotulo' in targets:
        y = targets['multirrotulo']
        colunas_falha = targets['colunas_multirrotulo']
        tipo_target = 'multirrotulo'
    else:
        st.error("❌ Dados não contêm colunas de falha necessárias para modelagem multirrótulo")
        st.stop()
elif tipo_modelagem.startswith("Binária"):
    if 'binario' in targets:
        y = targets['binario']
        tipo_target = 'binario'
    else:
        st.error("❌ Dados não contêm informações de falha necessárias para modelagem binária")
        st.stop()
else:  # Multiclasse
    if 'multiclasse' in targets:
        y = targets['multiclasse']
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        tipo_target = 'multiclasse'
    else:
        st.error("❌ Dados não contêm informações de falha necessárias para modelagem multiclasse")
        st.stop()

# Divisão treino/validação
if tipo_target == 'multirrotulo':
    X_treino, X_valid, y_treino, y_valid = train_test_split(
        X, y, test_size=percentual_teste/100, random_state=semente_aleatoria
    )
else:
    X_treino, X_valid, y_treino, y_valid = train_test_split(
        X, y, test_size=percentual_teste/100, random_state=semente_aleatoria,
        stratify=y if len(np.unique(y)) > 1 else None
    )

# Treinar modelo
with st.spinner("🔄 Treinando modelo..."):
    if tipo_target == 'multirrotulo':
        modelo, pred_valid, metricas_classes = treinar_modelo_multirrotulo(
            X_treino, y_treino, X_valid, y_valid, colunas_falha, n_estimators
        )
        
        # Mostrar métricas por classe
        st.subheader("📈 Métricas por Classe")
        cols = st.columns(len(colunas_falha))
        for i, (classe, acc) in enumerate(metricas_classes.items()):
            with cols[i]:
                st.metric(classe, f"{acc:.3f}")
                
    else:
        modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=semente_aleatoria, n_jobs=-1)
        modelo.fit(X_treino, y_treino)
        pred_valid = modelo.predict(X_valid)
        
        # Métricas gerais
        acc = accuracy_score(y_valid, pred_valid)
        st.metric("Acurácia de Validação", f"{acc:.4f}")
        
        with st.expander("📋 Relatório Detalhado"):
            st.text(classification_report(y_valid, pred_valid))

# Feature Importance
if hasattr(modelo, 'feature_importances_') or hasattr(modelo, 'estimators_'):
    st.subheader("📊 Importância das Features")
    
    if tipo_target == 'multirrotulo':
        # Para modelo multirrótulo, calcular importância média
        importancias = np.mean([est.feature_importances_ for est in modelo.estimators_], axis=0)
    else:
        importancias = modelo.feature_importances_
    
    df_importance = pd.DataFrame({
        'feature': features_numericas,
        'importance': importancias
    }).sort_values('importance', ascending=True)
    
    fig_imp = px.bar(
        df_importance.tail(10), 
        x='importance', 
        y='feature',
        orientation='h',
        title="Top 10 Features Mais Importantes"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ---------------- Predições no Conjunto de Teste ----------------
if dados_teste is not None:
    st.header("🎯 Predições no Conjunto de Teste")
    
    # Aplicar mesmas transformações aos dados de teste
    dados_teste_eng = criar_features_engenharia(
        dados_teste,
        incluir_dif_temp=usar_dif_temp,
        incluir_potencia=usar_potencia,
        incluir_eficiencia=usar_eficiencia
    )
    
    # Encoding da variável tipo se existir
    if 'tipo' in dados_teste_eng.columns and le_tipo is not None:
        dados_teste_eng['tipo_encoded'] = le_tipo.transform(dados_teste_eng['tipo'])
    
    # Selecionar apenas features disponíveis
    features_teste = [col for col in features_numericas if col in dados_teste_eng.columns]
    X_teste = dados_teste_eng[features_teste]
    
    # Aplicar normalização se foi usada
    if scaler is not None:
        X_teste = pd.DataFrame(scaler.transform(X_teste), columns=X_teste.columns, index=X_teste.index)
    
    # Fazer predições
    with st.spinner("🔮 Fazendo predições..."):
        if tipo_target == 'multirrotulo':
            pred_teste = modelo.predict(X_teste)
            
            # Criar DataFrame com predições para cada classe
            df_predicoes = pd.DataFrame(pred_teste, columns=colunas_falha)
            
        else:
            pred_teste = modelo.predict(X_teste)
            
            if tipo_target == 'binario':
                # Para modelo binário, criar predições para todas as classes
                # Assumindo que se há falha, é distribuída entre os tipos
                df_predicoes = pd.DataFrame(0, index=range(len(pred_teste)), 
                                          columns=['FDF', 'FDC', 'FP', 'FTE', 'FA'])
                
                # Estratégia simples: se predição é 1, marcar como FDF (mais comum)
                df_predicoes.loc[pred_teste == 1, 'FDF'] = 1
                
            else:  # multiclasse
                # Converter predições de volta para one-hot
                pred_classes = le_target.inverse_transform(pred_teste)
                df_predicoes = pd.DataFrame(0, index=range(len(pred_teste)),
                                          columns=['FDF', 'FDC', 'FP', 'FTE', 'FA'])
                
                for i, classe in enumerate(pred_classes):
                    if classe in ['FDF', 'FDC', 'FP', 'FTE', 'FA']:
                        df_predicoes.loc[i, classe] = 1
    
    # Mostrar estatísticas das predições
    st.subheader("📊 Estatísticas das Predições")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Amostras", len(df_predicoes))
    with col2:
        total_falhas = df_predicoes.sum().sum()
        st.metric("Total de Falhas Preditas", int(total_falhas))
    with col3:
        taxa_falha_pred = (df_predicoes.sum(axis=1) > 0).mean()
        st.metric("Taxa de Falha Predita", f"{taxa_falha_pred:.1%}")
    
    # Distribuição das predições
    contagens_pred = df_predicoes.sum().sort_values(ascending=True)
    fig_pred = px.bar(
        x=contagens_pred.values,
        y=contagens_pred.index,
        orientation='h',
        title="Distribuição das Predições por Tipo de Falha",
        labels={'x': 'Número de Predições', 'y': 'Tipo de Falha'}
    )
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Preview das predições
    st.subheader("👀 Preview das Predições")
    st.dataframe(df_predicoes.head(10))
    
    # ---------------- Geração do Arquivo CSV ----------------
    st.header("💾 Geração do Arquivo CSV para API")
    
    # Validar formato do CSV
    colunas_esperadas = ['FDF', 'FDC', 'FP', 'FTE', 'FA']
    if all(col in df_predicoes.columns for col in colunas_esperadas):
        
        # Garantir que são apenas 0s e 1s
        df_final = df_predicoes[colunas_esperadas].copy()
        for col in colunas_esperadas:
            df_final[col] = df_final[col].astype(int).clip(0, 1)
        
        # Estatísticas finais
        st.subheader("✅ Validação Final do CSV")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Linhas", len(df_final))
        with col2:
            st.metric("Colunas", len(df_final.columns))
        with col3:
            valores_unicos = set()
            for col in colunas_esperadas:
                valores_unicos.update(df_final[col].unique())
            st.metric("Valores Únicos", str(sorted(valores_unicos)))
        with col4:
            sem_falhas = (df_final.sum(axis=1) == 0).sum()
            st.metric("Amostras sem Falha", sem_falhas)
        
        # Mostrar preview do CSV final
        with st.expander("📋 Preview do CSV Final"):
            st.dataframe(df_final.head(20))
            
            # Estatísticas por coluna
            st.write("**Estatísticas por Tipo de Falha:**")
            stats_df = pd.DataFrame({
                'Tipo_Falha': colunas_esperadas,
                'Total_Predicoes': [df_final[col].sum() for col in colunas_esperadas],
                'Percentual': [f"{df_final[col].mean():.1%}" for col in colunas_esperadas]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        # Botão para download
        csv_string = df_final.to_csv(index=False)
        
        st.download_button(
            label="📥 Baixar CSV para API",
            data=csv_string,
            file_name="bootcamp_predictions.csv",
            mime="text/csv",
            help="Arquivo CSV formatado para envio à API de avaliação"
        )
        
        # Informações sobre o arquivo
        st.info(f"""
        📋 **Informações do Arquivo:**
        - ✅ Formato: CSV com colunas FDF, FDC, FP, FTE, FA
        - ✅ Valores: Apenas 0 e 1 (binários)
        - ✅ Tamanho: {len(df_final)} linhas × {len(df_final.columns)} colunas
        - ✅ Encoding: UTF-8
        - ✅ Pronto para API de avaliação
        """)
        
        # Seção opcional para API (se quiser testar)
        with st.expander("🌐 Teste Opcional com API Externa"):
            st.write("Se você quiser testar o arquivo diretamente:")
            
            email_api = st.text_input("Email para API:", placeholder="seu-email@exemplo.com")
            senha_api = st.text_input("Senha:", type="password")
            threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05)
            
            if st.button("🧪 Testar na API"):
                if email_api and senha_api:
                    try:
                        # Endpoint de registro
                        resp_register = requests.post(
                            "http://34.193.187.218:5000/users/register",
                            json={"email": email_api, "password": senha_api},
                            timeout=30
                        )
                        
                        if resp_register.status_code == 200:
                            token_data = resp_register.json()
                            if 'token' in token_data:
                                token = token_data['token']
                                st.success("✅ Token obtido com sucesso!")
                                
                                # Enviar predições
                                headers = {"X-API-Key": token}
                                params = {"threshold": threshold}
                                files = {"file": ("submission.csv", csv_string, "text/csv")}
                                
                                resp_eval = requests.post(
                                    "http://34.193.187.218:5000/evaluate/multilabel_metrics",
                                    headers=headers,
                                    files=files,
                                    params=params,
                                    timeout=60
                                )
                                
                                st.write(f"Status da avaliação: {resp_eval.status_code}")
                                if resp_eval.status_code == 200:
                                    resultados = resp_eval.json()
                                    st.json(resultados)
                                else:
                                    st.error(f"Erro na avaliação: {resp_eval.text}")
                            else:
                                st.error("Token não encontrado na resposta")
                        else:
                            st.error(f"Erro no registro: {resp_register.status_code}")
                            
                    except Exception as e:
                        st.error(f"Erro na comunicação: {e}")
                else:
                    st.warning("Preencha email e senha para testar")
        
    else:
        st.error("❌ Erro na geração das predições. Verifique os dados de entrada.")

else:
    st.info("ℹ️ Carregue o arquivo Bootcamp_test.csv para gerar predições e o arquivo CSV para a API")

# ---------------- Rodapé ----------------
st.markdown("---")
st.markdown("**Desenvolvido para o Bootcamp de Ciência de Dados e IA** 🚀")

