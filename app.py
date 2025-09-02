import streamlit as st
import io

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
        # Lê o arquivo como texto para demonstração
        conteudo = arquivo.getvalue().decode('utf-8')
        linhas = conteudo.split('\n')
        st.sidebar.success(f"✅ Arquivo carregado: {len(linhas)} linhas")
        
        # Mostra preview
        st.sidebar.subheader("👀 Preview dos Dados")
        st.sidebar.text("\n".join(linhas[:5]))
        
    except Exception as e:
        st.sidebar.error(f"❌ Erro: {e}")

# Abas principais
tab1, tab2 = st.tabs(["📊 Análise", "🔮 Previsão"])

with tab1:
    st.header("Análise dos Dados")
    
    if arquivo:
        st.info("ℹ️ Funcionalidade de análise requer pandas/numpy")
        st.write("Para análise completa, instale localmente:")
        st.code("pip install pandas numpy plotly")
    else:
        st.warning("⚠️ Faça upload de um arquivo CSV")

with tab2:
    st.header("Fazer Previsões")
    
    # Previsão manual
    st.subheader("Previsão Individual")
    
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
                import random
                resultado = "✅ BOM" if random.random() > 0.7 else "⚠️ ATENÇÃO"
                probabilidade = f"{random.randint(10, 90)}%"
                st.write(f"**{falha}**: {resultado} ({probabilidade})")

# Seção de informações
st.markdown("---")
st.header("ℹ️ Informações do Projeto")

st.write("""
Este é um projeto demonstrativo de Data Science para previsão de falhas em máquinas industriais.

**Funcionalidades completas disponíveis localmente:**
```bash
pip install streamlit pandas numpy scikit-learn plotly
