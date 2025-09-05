# app.py
import streamlit as st
import traceback, sys
import pandas as pd
from utils import carregar_e_processar_dados, preprocess_pipeline
from modelos import comparar_e_treinar_modelos, gerar_predicoes_para_submissao

st.set_page_config(page_title="🔧 Manutenção Preditiva", page_icon="🔧", layout="wide")
st.title("🔧 Sistema Inteligente de Manutenção Preditiva")

st.markdown("""
Carregue o arquivo de treino (obrigatório) e o arquivo de teste (opcional).
O app fará pré-processamento, comparará modelos (RandomForest, LightGBM, XGBoost quando disponíveis)
e permitirá exportar predições no formato esperado pela API do Bootcamp.
""")

arquivo_treino = st.file_uploader("📂 Selecione Bootcamp_train.csv", type=["csv"])
arquivo_teste = st.file_uploader("📂 Selecione Bootcamp_test.csv (opcional)", type=["csv"])

if arquivo_treino:
    try:
        treino_df = carregar_e_processar_dados(arquivo_treino)
        teste_df = carregar_e_processar_dados(arquivo_teste) if arquivo_teste else None

        st.success("✅ Arquivo(s) carregado(s)")
        st.write("Preview treino:")
        st.dataframe(treino_df.head())

        with st.expander("⚙️ Configurações de pré-processamento"):
            st.write("Targets padrão: fdf, fdc, fp, fte, fa")
            threshold = st.slider("Threshold de decisão (para modelos probabilísticos)", 0.0, 1.0, 0.5)

        st.info("Iniciando pré-processamento e comparação de modelos. Isso pode levar alguns minutos.")

        X_train, X_val, y_train, y_val, X_test_proc, preprocess_objects, feature_names, targets = preprocess_pipeline(
            treino_df, teste_df, verbose=True
        )

        st.write("Dimensões:", X_train.shape, X_val.shape)
        st.write("Features usadas:", feature_names)
        st.write("Targets:", targets)

        # Treinar e comparar modelos
        results, best_model = comparar_e_treinar_modelos(X_train, y_train, X_val, y_val)

        st.write("## ✅ Resultados da comparação")
        st.dataframe(pd.DataFrame(results).sort_values(by="mean_f1", ascending=False).reset_index(drop=True))

        st.success(f"Melhor modelo: {results[0]['model_name']} (ver primeira linha da tabela)")

        if X_test_proc is not None:
            st.subheader("📊 Gerar predições para o conjunto de teste")
            df_pred = gerar_predicoes_para_submissao(best_model, X_test_proc, targets, original_test_df=teste_df)
            st.write("Amostra das predições:")
            st.dataframe(df_pred.head())
            csv = df_pred.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Baixar predições (.csv)", csv, "predicoes_submission.csv", "text/csv")
        else:
            st.info("Nenhum arquivo de teste fornecido — você pode fazer upload ou usar 'train_and_select.py' localmente depois.")

    except Exception as e:
        tb = traceback.format_exc()
        st.error("❌ Erro no processamento")
        st.code(tb)
        print(tb, file=sys.stderr)
else:
    st.info("Faça upload do Bootcamp_train.csv para começar.")
