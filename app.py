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

# trecho relevante de app.py (substitua a lógica depois do upload)
from utils import carregar_e_processar_dados, detect_failure_columns, preprocess_pipeline
from modelos import comparar_e_treinar_modelos, gerar_predicoes_para_submissao

arquivo_treino = st.file_uploader("📂 Selecione Bootcamp_train.csv", type=["csv"])
arquivo_teste = st.file_uploader("📂 Selecione Bootcamp_test.csv (opcional)", type=["csv"])

if arquivo_treino:
    treino_df = carregar_e_processar_dados(arquivo_treino)
    teste_df = carregar_e_processar_dados(arquivo_teste) if arquivo_teste else None

    st.success("✅ Arquivo de treino carregado")
    st.write("Preview (treino):")
    st.dataframe(treino_df.head())

    # 1) Detectar automaticamente e mostrar opções ao usuário
    detection = detect_failure_columns(treino_df)
    st.write("✅ Detecção automática (tentativa):")
    st.json(detection["detected_map"])
    st.write("Colunas binárias candidatas:", detection["candidate_binary_cols"])

    st.markdown("**Confirme o mapeamento das colunas de falha** (ou escolha manualmente):")
    canonical = ["fdf", "fdc", "fp", "fte", "fa"]
    # build options
    cols_options = [""] + detection["all_columns"]
    user_map = {}
    for can in canonical:
        default = detection["detected_map"].get(can) if detection["detected_map"].get(can) else ""
        user_map[can] = st.selectbox(f"Coluna para {can.upper()}", options=cols_options, index=cols_options.index(default) if default in cols_options else 0, key=f"map_{can}")

    if st.button("✅ Confirmar mapeamento e rodar pré-processamento"):
        # verify all selected
        if not all(user_map.values()):
            st.error("Selecione todas as 5 colunas de falha antes de continuar.")
        else:
            # preparar lista na ordem canônica
            target_columns_actual = [user_map[c] for c in canonical]

            # chamar pipeline passando os nomes reais
            X_train, X_val, y_train, y_val, X_test_proc, preprocess_objects, features, detected = preprocess_pipeline(
                treino_df, teste_df, target_columns=target_columns_actual, verbose=True
            )

            st.success("Pré-processamento concluído")
            st.write(f"Features: {len(features)} colunas")
            st.write("Targets (colunas reais):", detected)

            # treinar/comparar modelos
            results, best_model = comparar_e_treinar_modelos(X_train, y_train, X_val, y_val)
            st.write("Resultados comparativos:")
            st.dataframe(pd.DataFrame(results).sort_values("mean_f1", ascending=False))

            if X_test_proc is not None:
                df_pred = gerar_predicoes_para_submissao(best_model, X_test_proc, preprocess_objects["canonical_targets"], target_map=preprocess_objects["target_columns_actual"], original_test_df=teste_df)
                st.write("Amostra das predições:")
                st.dataframe(df_pred.head())
                st.download_button("📥 Baixar predições (.csv)", df_pred.to_csv(index=False).encode("utf-8"), "predicoes_submission.csv", "text/csv")
            else:
                st.info("Nenhum arquivo de teste fornecido — você pode fazer upload ou usar 'train_and_select.py' localmente depois.")


