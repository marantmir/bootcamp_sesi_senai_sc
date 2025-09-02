import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configuração básica
st.set_page_config(page_title="Previsão de Falhas", layout="wide")

st.title("🔧 Sistema de Previsão de Falhas")
st.write("Projeto de Data Science para prever falhas em máquinas industriais")

# Listas
FALHAS = ["FDF", "FDC", "FP", "FTE", "FA"]
COLUNAS = ["tipo", "temperatura_ar", "temperatura_processo", "umidade_relativa",
           "velocidade_rotacional", "torque", "desgaste_ferramenta"]

# Sidebar - Upload de dados
st.sidebar.header("📂 Carregar Dados")
arquivo = st.sidebar.file_uploader("Selecione arquivo CSV", type=["csv"])

dados = None
if arquivo:
    try:
        dados = pd.read_csv(arquivo)
        st.sidebar.success(f"✅ Dados carregados: {len(dados)} linhas")
    except Exception as e:
        st.sidebar.error(f"❌ Erro: {e}")

# Tenta carregar dados locais
if dados is None:
    try:
        dados = pd.read_csv("data/Bootcamp_train.csv")
        st.sidebar.info("📂 Usando dados locais")
    except:
        st.warning("⚠️ Faça upload de um arquivo CSV")

if dados is None:
    st.stop()

# Limpeza básica
dados_limpos = dados.copy()
dados_limpos.replace("?", np.nan, inplace=True)

# Converte colunas numéricas
for col in COLUNAS:
    if col in dados_limpos.columns and col != 'tipo':
        dados_limpos[col] = pd.to_numeric(dados_limpos[col], errors="coerce")

# Sidebar - Preview
st.sidebar.subheader("👀 Preview dos Dados")
st.sidebar.dataframe(dados_limpos.head(3))

# Abas principais
tab1, tab2 = st.tabs(["📊 Análise", "🔮 Previsão"])

with tab1:
    st.header("Análise dos Dados")
    
    # Verifica se tem dados de falha
    tem_falhas = all(f in dados_limpos.columns for f in FALHAS)
    
    if tem_falhas:
        st.subheader("Distribuição de Falhas")
        contagem = dados_limpos[FALHAS].sum()
        fig = px.bar(x=contagem.index, y=contagem.values, 
                    labels={'x': 'Tipo de Falha', 'y': 'Quantidade'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Arquivo de teste - sem dados de falha")
    
    # Histogramas
    st.subheader("Distribuição das Variáveis")
    colunas_numericas = [col for col in COLUNAS if col in dados_limpos.columns and col != 'tipo']
    
    if colunas_numericas:
        variavel = st.selectbox("Selecione variável:", colunas_numericas)
        fig_hist = px.histogram(dados_limpos, x=variavel, title=f"Distribuição de {variavel}")
        st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.header("Fazer Previsões")
    st.info("ℹ️ Funcionalidade de previsão requer modelo treinado")
    
    # Demonstração simples
    st.subheader("Simulação de Previsão")
    
    with st.form("form_previsao"):
        col1, col2 = st.columns(2)
        
        with col1:
            tipo = st.selectbox("Tipo", ["L", "M", "H"])
            temp_ar = st.number_input("Temp. Ar (K)", value=300.0)
            temp_processo = st.number_input("Temp. Processo (K)", value=310.0)
            umidade = st.number_input("Umidade (%)", value=45.0)
        
        with col2:
            velocidade = st.number_input("Velocidade (RPM)", value=1500.0)
            torque = st.number_input("Torque (Nm)", value=40.0)
            desgaste = st.number_input("Desgaste (min)", value=120.0)
        
        if st.form_submit_button("🎯 Simular Previsão"):
            st.success("✅ Simulação concluída!")
            
            # Simulação simples
            st.write("**Resultados simulados:**")
            for falha in FALHAS:
                # Simulação aleatória para demonstração
                resultado = "✅ BOM" if np.random.random() > 0.7 else "⚠️ ATENÇÃO"
                st.write(f"**{falha}**: {resultado}")

# Footer
st.markdown("---")
st.write("Desenvolvido como projeto de aprendizado em Data Science")
