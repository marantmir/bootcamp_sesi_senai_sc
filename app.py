import streamlit as st
import random

# Configuração básica
st.set_page_config(page_title="Previsão de Falhas", layout="wide")

st.title("🔧 Sistema de Previsão de Falhas")
st.write("Projeto de Data Science para prever falhas em máquinas industriais")

# Listas
FALHAS = ["FDF", "FDC", "FP", "FTE", "FA"]

# Sidebar - Upload de dados
st.sidebar.header("📂 Carregar Dados")
arquivo = st.sidebar.file_uploader("Selecione arquivo CSV", type=["csv"])

if arquivo:
    try:
        # Lê o arquivo como texto para demonstração
        conteudo = arquivo.getvalue().decode('utf-8')
        linhas = conteudo.split('\n')
        st.sidebar.success(f"✅ Arquivo carregado: {len(linhas)} linhas")
        
        # Mostra preview
        st.sidebar.subheader("👀 Preview dos Dados")
        for i, linha in enumerate(linhas[:5]):
            if i < 5:  # Mostra apenas as primeiras 5 linhas
                st.sidebar.text(linha[:100] + "..." if len(linha) > 100 else linha)
                
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao ler arquivo")

# Abas principais
tab1, tab2, tab3 = st.tabs(["📊 Análise", "🔮 Previsão", "ℹ️ Sobre"])

with tab1:
    st.header("Análise dos Dados")
    
    if arquivo:
        st.info("""
        ℹ️ **Funcionalidade de análise completa disponível localmente**
        
        Para usar todas as funcionalidades de análise, instale:
        ```bash
        pip install pandas numpy plotly
        ```
        
        **Recursos disponíveis localmente:**
        - Visualização de dados interativa
        - Gráficos e histogramas
        - Análise estatística
        - Distribuição de falhas
        """)
        
        # Simulação simples de análise
        st.subheader("Simulação de Análise")
        st.write("Com dados completos, você veria:")
        st.write("✅ Distribuição de tipos de falha")
        st.write("✅ Histogramas das variáveis numéricas")
        st.write("✅ Correlação entre sensores")
        st.write("✅ Estatísticas descritivas")
        
    else:
        st.warning("⚠️ Faça upload de um arquivo CSV para ver análise")

with tab2:
    st.header("Fazer Previsões")
    
    # Previsão manual
    st.subheader("Previsão Individual")
    
    with st.form("form_previsao"):
        col1, col2 = st.columns(2)
        
        with col1:
            tipo = st.selectbox("Tipo da Máquina", ["L", "M", "H"])
            temp_ar = st.number_input("Temperatura do Ar (K)", value=300.0, min_value=0.0)
            temp_processo = st.number_input("Temperatura do Processo (K)", value=310.0, min_value=0.0)
            umidade = st.number_input("Umidade Relativa (%)", value=45.0, min_value=0.0, max_value=100.0)
        
        with col2:
            velocidade = st.number_input("Velocidade Rotacional (RPM)", value=1500.0, min_value=0.0)
            torque = st.number_input("Torque (Nm)", value=40.0, min_value=0.0)
            desgaste = st.number_input("Desgaste da Ferramenta (min)", value=120.0, min_value=0.0)
        
        if st.form_submit_button("🎯 Fazer Previsão"):
            st.success("✅ Previsão simulada concluída!")
            
            # Simulação de resultados
            st.subheader("Resultados da Previsão:")
            
            for falha in FALHAS:
                # Gera resultado aleatório para demonstração
                tem_falha = random.random() > 0.7
                probabilidade = random.randint(10, 95)
                
                if tem_falha:
                    st.error(f"**{falha}**: ❌ RISCO DE FALHA ({probabilidade}% de chance)")
                else:
                    st.success(f"**{falha}**: ✅ NORMAL ({probabilidade}% de confiança)")
            
            st.info("""
            💡 **Nota:** Esta é uma simulação. 
            Para previsões reais com machine learning, instale localmente:
            ```bash
            pip install scikit-learn pandas numpy
            ```
            """)

with tab3:
    st.header("Sobre o Projeto")
    
    st.info("""
    ## 🔧 Sistema de Previsão de Falhas
    
    **Objetivo:** Prever falhas em máquinas industriais usando machine learning
    
    **Funcionalidades completas disponíveis localmente:**
    - Análise exploratória de dados
    - Visualizações interativas
    - Modelos de machine learning
    - Previsões em tempo real
    
    **Tecnologias utilizadas:**
    - Python 🐍
    - Streamlit 🎈
    - Scikit-learn 🤖
    - Pandas 🐼
    - Plotly 📊
    - NumPy 🔢
    """)
    
    st.subheader("🚀 Como Executar Localmente")
    
    st.code("""
# Clone o repositório
git clone [seu-repositorio]
cd [pasta-do-projeto]

# Instale as dependências
pip install streamlit pandas numpy scikit-learn plotly

# Execute o app
streamlit run app.py
    """)
    
    st.subheader("📋 Estrutura do Projeto")
    st.write("""
    ```
    projeto/
    ├── app.py              # Aplicação principal
    ├── requirements.txt    # Dependências
    ├── data/              # Dados de treino/teste
    └── modelos/           # Modelos treinados
    ```
    """)
    
    st.subheader("📞 Contato")
    st.write("Desenvolvido como projeto de aprendizado em Data Science")
    st.write("📧 Email: marcoantoniomiranda713@gmail.com")

# Footer
st.markdown("---")
st.caption("© 2025 - Projeto de Data Science | Desenvolvido para aprendizado")

