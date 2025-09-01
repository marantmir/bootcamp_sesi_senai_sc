"""
App Streamlit para o projeto de manutenção preditiva
"""

import streamlit as st
import pandas as pd
import os
import plotly.express as px
from utilitarios import carregar_modelo_treinado, limpar_dados, FALHAS, carregar_dados, COLUNAS_USAR

# Configuração básica da página
st.set_page_config(
    page_title="Sistema de Previsão de Falhas", 
    layout="wide",
    page_icon="🔧"
)

# Título bonito
st.title("🔧 Sistema de Previsão de Falhas")
st.write("""
Projeto desenvolvido durante o bootcamp de Data Science.
Usa machine learning para prever falhas em máquinas industriais.
""")

# Inicializa as variáveis de sessão
if 'modelo_carregado' not in st.session_state:
    st.session_state.modelo_carregado = None
    st.session_state.ultimo_modelo_carregado = ""

if 'dados_atuais' not in st.session_state:
    st.session_state.dados_atuais = None

# Sidebar - configurações
st.sidebar.header("⚙️ Configurações")

# Upload de arquivo
st.sidebar.subheader("📁 Dados")
arquivo_up = st.sidebar.file_uploader(
    "Escolha um arquivo CSV", 
    type=["csv"],
    help="Pode ser arquivo de treino ou teste"
)

if arquivo_up:
    try:
        dados = pd.read_csv(arquivo_up)
        st.session_state.dados_atuais = dados
        st.sidebar.success(f"✅ Carregado: {len(dados)} linhas")
    except Exception as e:
        st.sidebar.error(f"❌ Erro: {e}")

# Se não fez upload, tenta carregar local
if st.session_state.dados_atuais is None:
    locais_tentar = [
        "data/Bootcamp_train.csv",
        "Bootcamp_train.csv", 
        "train.csv",
        "dados_treino.csv"
    ]
    
    for local in locais_tentar:
        if os.path.exists(local):
            dados = carregar_dados(local)
            if dados is not None:
                st.session_state.dados_atuais = dados
                st.sidebar.info(f"📂 Usando arquivo local: {local}")
                break

if st.session_state.dados_atuais is None:
    st.warning("""
    ⚠️ Não encontrei nenhum arquivo de dados.
    
    Por favor:
    1. Faça upload de um CSV, ou
    2. Coloque um arquivo chamado 'Bootcamp_train.csv' na pasta 'data/'
    """)
    st.stop()

# Processa os dados
dados_limpos = limpar_dados(st.session_state.dados_atuais)

# Mostra preview na sidebar
st.sidebar.subheader("👀 Preview dos Dados")
st.sidebar.dataframe(dados_limpos.head(3))

# Carregamento do modelo
st.sidebar.subheader("🤖 Modelo")
caminho_modelo = st.sidebar.text_input(
    "Caminho do modelo treinado", 
    value="modelos/modelo_treinado.joblib",
    help="Caminho relativo para o arquivo .joblib"
)

btn_carregar = st.sidebar.button("🔄 Carregar Modelo")

if btn_carregar:
    with st.spinner("Carregando modelo..."):
        modelo = carregar_modelo_treinado(caminho_modelo)
        if modelo is not None:
            st.session_state.modelo_carregado = modelo
            st.session_state.ultimo_modelo_carregado = caminho_modelo
            st.sidebar.success("✅ Modelo carregado!")
        else:
            st.sidebar.error("❌ Não consegui carregar o modelo")

# Mostra status do modelo
if st.session_state.modelo_carregado:
    st.sidebar.info(f"📦 Modelo: {st.session_state.ultimo_modelo_carregado}")
else:
    st.sidebar.warning("⚠️ Nenhum modelo carregado")

# Abas principais
tab1, tab2, tab3 = st.tabs(["📊 Análise", "🔮 Previsão", "❓ Ajuda"])

with tab1:
    st.header("Análise Exploratória")
    
    # Verifica se tem dados de falha
    tem_falhas = all(f in dados_limpos.columns for f in FALHAS)
    
    if tem_falhas:
        st.subheader("Distribuição das Falhas")
        
        # Calcula totais
        totais = dados_limpos[FALHAS].sum()
        
        # Gráfico de barras
        fig = px.bar(
            x=totais.index,
            y=totais.values,
            labels={'x': 'Tipo de Falha', 'y': 'Quantidade'},
            title="Quantidade de cada tipo de falha"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostra números também
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total FDF", totais['FDF'])
        with col2:
            st.metric("Total FDC", totais['FDC'])
        with col3:
            st.metric("Total FP", totais['FP'])
        
        col4, col5 = st.columns(2)
        with col4:
            st.metric("Total FTE", totais['FTE'])
        with col5:
            st.metric("Total FA", totais['FA'])
    else:
        st.info("""
        ℹ️ Este arquivo parece ser de teste - não contém informações de falhas.
        Use a aba de Previsão para fazer previsões com estes dados.
        """)
    
    # Histogramas interativos
    st.subheader("Distribuição das Variáveis")
    
    # Pega só colunas numéricas
    colunas_numericas = [col for col in COLUNAS_USAR if col in dados_limpos.columns and col != 'tipo']
    
    if colunas_numericas:
        variavel_escolhida = st.selectbox(
            "Escolha uma variável para ver distribuição:",
            colunas_numericas
        )
        
        fig_hist = px.histogram(
            dados_limpos, 
            x=variavel_escolhida,
            title=f"Distribuição de {variavel_escolhida}",
            nbins=30
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Estatísticas básicas
        stats = dados_limpos[variavel_escolhida].describe()
        st.write("**Estatísticas:**")
        st.write(f"Média: {stats['mean']:.2f} | Mediana: {stats['50%']:.2f}")
        st.write(f"Mín: {stats['min']:.2f} | Máx: {stats['max']:.2f}")
    else:
        st.warning("Não encontrei variáveis numéricas para mostrar")

with tab2:
    st.header("Fazer Previsões")
    
    # Previsão manual - formulário
    st.subheader("Previsão Individual")
    st.write("Preencha os dados da máquina para prever falhas:")
    
    with st.form("form_previsao_manual"):
        col1, col2 = st.columns(2)
        
        with col1:
            tipo = st.selectbox("Tipo da Máquina", ["L", "M", "H"], help="L, M ou H")
            temp_ar = st.number_input("Temperatura do Ar (K)", value=298.0, min_value=0.0, step=0.1)
            temp_processo = st.number_input("Temperatura do Processo (K)", value=308.0, min_value=0.0, step=0.1)
            umidade = st.number_input("Umidade Relativa (%)", value=45.0, min_value=0.0, max_value=100.0, step=0.1)
        
        with col2:
            rotacao = st.number_input("Velocidade Rotacional (RPM)", value=1550.0, min_value=0.0, step=1.0)
            torque = st.number_input("Torque (Nm)", value=42.0, min_value=0.0, step=0.1)
            desgaste = st.number_input("Desgaste da Ferramenta (min)", value=120.0, min_value=0.0, step=1.0)
        
        btn_prever = st.form_submit_button("🎯 Fazer Previsão")
        
        if btn_prever:
            if st.session_state.modelo_carregado is None:
                st.error("""
                ❌ Preciso de um modelo treinado!
                
                Por favor:
                1. Treine um modelo com treinar.py
                2. Carregue o modelo na sidebar
                """)
            else:
                try:
                    # Prepara os dados no formato certo
                    dados_input = pd.DataFrame([{
                        'tipo': tipo,
                        'temperatura_ar': temp_ar,
                        'temperatura_processo': temp_processo,
                        'umidade_relativa': umidade,
                        'velocidade_rotacional': rotacao,
                        'torque': torque,
                        'desgaste_ferramenta': desgaste
                    }])
                    
                    # Faz a previsão
                    previsao = st.session_state.modelo_carregado.predict(dados_input)
                    probabilidades = st.session_state.modelo_carregado.predict_proba(dados_input)
                    
                    st.success("✅ Previsão concluída!")
                    
                    # Mostra os resultados
                    st.subheader("Resultados:")
                    
                    for i, falha in enumerate(FALHAS):
                        # Tenta pegar a probabilidade - pode variar conforme o modelo
                        try:
                            prob = probabilidades[i][0][1]  # Para alguns modelos
                        except:
                            try:
                                prob = probabilidades[0][i]  # Para outros
                            except:
                                prob = 0.5
                        
                        # Formata bonito
                        predicao = previsao[0][i]
                        cor = "🔴" if predicao == 1 else "🟢"
                        texto = "SIM" if predicao == 1 else "NÃO"
                        
                        st.write(f"""
                        **{falha}**: {cor} **{texto}** 
                        *({prob*100:.1f}% de chance)*
                        """)
                    
                except Exception as e:
                    st.error(f"❌ Erro na previsão: {str(e)}")
                    st.info("""
                    💡 Dica: Verifique se o modelo foi treinado com as mesmas 
                    colunas que está tentando prever.
                    """)
    
    # Previsão em lote
    st.subheader("Previsão em Arquivo")
    st.write("Faça previsões para um arquivo CSV completo:")
    
    arquivo_previsao = st.file_uploader(
        "Selecione arquivo CSV para previsão",
        type=["csv"],
        key="arquivo_previsao"
    )
    
    if arquivo_previsao and st.session_state.modelo_carregado:
        if st.button("📊 Processar Arquivo Inteiro"):
            with st.spinner("Processando... pode demorar um pouco"):
                try:
                    dados_previsao = pd.read_csv(arquivo_previsao)
                    dados_limpos_previsao = limpar_dados(dados_previsao)
                    
                    # Pega só as colunas que o modelo precisa
                    X_previsao = dados_limpos_previsao[COLUNAS_USAR]
                    
                    # Faz as previsões
                    previsoes = st.session_state.modelo_carregado.predict(X_previsao)
                    probs = st.session_state.modelo_carregado.predict_proba(X_previsao)
                    
                    # Monta o resultado
                    resultado = pd.DataFrame()
                    
                    # Mantém o ID se existir
                    if 'id' in dados_limpos_previsao.columns:
                        resultado['id'] = dados_limpos_previsao['id']
                    
                    # Adiciona as previsões
                    for i, falha in enumerate(FALHAS):
                        resultado[falha] = [int(linha[i]) for linha in previsoes]
                        
                        # Tenta pegar probabilidades
                        try:
                            probs_col = [p[i][1] for p in probs]
                        except:
                            try:
                                probs_col = [p[i] for p in probs]
                            except:
                                probs_col = [0.0] * len(resultado)
                        
                        resultado[f'prob_{falha}'] = probs_col
                    
                    st.success(f"✅ Processado! {len(resultado)} previsões")
                    
                    # Mostra preview
                    st.dataframe(resultado.head(10))
                    
                    # Botão para download
                    csv = resultado.to_csv(index=False)
                    st.download_button(
                        "💾 Baixar Resultados CSV",
                        csv,
                        "previsoes.csv",
                        "text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Erro ao processar arquivo: {str(e)}")

with tab3:
    st.header("❓ Ajuda e Informações")
    
    st.info("""
    ### Sobre este Projeto
    
    Este é meu projeto final do bootcamp de Data Science!
    
    **O que faz:**
    - Analisa dados de sensores de máquinas industriais
    - Prevê 5 tipos diferentes de falhas
    - Gera relatórios e previsões
    
    **Como usar:**
    1. **Treinar modelo**: Execute `python treinar.py`
    2. **Usar app**: Execute `streamlit run app.py`
    3. **Carregue dados**: CSV com as colunas certas
    4. **Faça previsões**: Individual ou em lote
    
    **Tecnologias usadas:**
    - Python 🐍
    - Scikit-learn 🤖
    - Streamlit 🎈
    - Pandas 🐼
    
    """)
    
    st.write("---")
    
    st.warning("""
    ⚠️ **Aviso importante:**
    Este é um projeto educacional. 
    Não use previsões para decisões reais de manutenção!
    """)
    
    st.write("---")
    st.write("📧 **Contato:** marcoantoniomiranda13@gmail.com")
    st.write("📅 **Última atualização:** Novembro 2023")