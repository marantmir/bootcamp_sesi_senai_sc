import io
import numpy as np
import pandas as pd
import requests
import streamlit as st

from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier

import plotly.express as px

# ---------------- Configuração ----------------
st.set_page_config(page_title="Sistema de Manutenção Preditiva", page_icon="🔧", layout="wide")
st.title("🔧 Sistema Inteligente de Manutenção Preditiva (versão corrigida)")

# ----- constantes (todas em lowercase para consistência) -----
COLS_FALHA = ["fdf", "fdc", "fp", "fte", "fa"]
COL_ALVO_BINARIA = "falha_maquina"

# ---------------- utilitários ----------------
def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    )
    return df

def coagir_e_reportar_numericos(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str,int]]:
    df = df.copy()
    rel = {}
    for c in cols:
        if c in df.columns:
            antes = df[c].isna().sum()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            depois = df[c].isna().sum()
            rel[c] = max(0, depois - antes)
    return df, rel

def engenhar_features(df: pd.DataFrame, usar_dif_temp=True, usar_potencia=True, usar_eficiencia=True) -> pd.DataFrame:
    df = df.copy()
    if usar_dif_temp and {"temperatura_processo","temperatura_ar"}.issubset(df.columns):
        df["dif_temperatura"] = df["temperatura_processo"] - df["temperatura_ar"]
        df["razao_temperatura"] = df["temperatura_processo"] / (df["temperatura_ar"] + 1e-6)
    if usar_potencia and {"torque","velocidade_rotacional"}.issubset(df.columns):
        df["potencia_kw"] = (df["torque"] * df["velocidade_rotacional"]) / 1000.0
    if usar_eficiencia and {"torque","desgaste_da_ferramenta"}.issubset(df.columns):
        df["eficiencia_ferramenta"] = df["torque"] / (df["desgaste_da_ferramenta"] + 1.0)
    return df

def preparar_targets(df: pd.DataFrame) -> Dict[str, object]:
    targets = {}
    tem_multilabel = all(c in df.columns for c in COLS_FALHA)
    tem_binario = COL_ALVO_BINARIA in df.columns
    if tem_multilabel:
        targets["multirrotulo"] = df[COLS_FALHA].fillna(0).astype(int).values
        targets["colunas_multirrotulo"] = COLS_FALHA
        if tem_binario:
            targets["binario"] = pd.to_numeric(df[COL_ALVO_BINARIA], errors="coerce").fillna(0).astype(int)
        else:
            targets["binario"] = df[COLS_FALHA].max(axis=1).astype(int)
        def tipo_principal(row):
            ativos = [c for c in COLS_FALHA if row[c] == 1]
            if len(ativos) == 0:
                return "sem_falha"
            if len(ativos) == 1:
                return ativos[0]
            return "multiplas_falhas"
        targets["multiclasse"] = df[COLS_FALHA].apply(tipo_principal, axis=1)
    else:
        if tem_binario:
            targets["binario"] = pd.to_numeric(df[COL_ALVO_BINARIA], errors="coerce").fillna(0).astype(int)
    return targets

def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).transpose()

def matriz_confusao_fig(y_true, y_pred, titulo="Matriz de Confusão"):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, aspect="auto", title=titulo, color_continuous_scale="Blues")
    fig.update_xaxes(title="Predito")
    fig.update_yaxes(title="Real")
    return fig

def preparar_X_dataframe(X_df: pd.DataFrame, referencia_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    - Converte categóricas via get_dummies
    - Coerce numeric nas que restarem
    - Preenche NaNs com mediana
    - Se referencia_cols for fornecido, reindexa para alinhar colunas do treino/teste
    Retorna (X_processed, lista_de_colunas_final)
    """
    X = X_df.copy()
    if X.empty:
        return X, []
    # identificar colunas categóricas (object/category)
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    # ainda pode ter colunas não-numéricas -> coerce
    non_num = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if non_num:
        for c in non_num:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # imputar com mediana para numéricas
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    # se referência fornecida, alinha
    if referencia_cols is not None:
        X = X.reindex(columns=referencia_cols, fill_value=0)
    return X, X.columns.tolist()

# ---------------- Sidebar / Inputs ----------------
with st.sidebar:
    st.header("⚙️ Configurações")
    arquivo_treino = st.file_uploader("📁 Bootcamp_train.csv", type=["csv"])
    arquivo_teste = st.file_uploader("📁 Bootcamp_test.csv (opcional)", type=["csv"])
    tipo_modelagem = st.selectbox("Tipo de Modelagem:", ["Multirrótulo", "Binária", "Multiclasse"])
    percentual_teste = st.slider("% Validação", 10, 40, 20, 5)
    semente = st.slider("Semente Aleatória", 0, 9999, 42)
    n_estimators = st.slider("Número de Árvores", 50, 600, 200, 50)
    usar_dif_temp = st.checkbox("Diferença temperatura", value=True)
    usar_potencia = st.checkbox("Potência (torque×velocidade)", value=True)
    usar_eficiencia = st.checkbox("Eficiência ferramenta", value=True)
    normalizar = st.checkbox("Normalizar (StandardScaler)", value=False)
    usar_proba = st.checkbox("Usar probabilidades ao gerar CSV (se disponível)", value=True)
    threshold_api = st.slider("Threshold para API", 0.0, 1.0, 0.5, 0.05)

# ---------------- Carregamento ----------------
@st.cache_data
def carregar_csv(f):
    return pd.read_csv(f)

dados_treino, dados_teste = None, None
mensagens = []
if arquivo_treino:
    try:
        dados_treino = carregar_csv(arquivo_treino)
        dados_treino = normalizar_colunas(dados_treino)
        mensagens.append(("success", f"Treino carregado: {dados_treino.shape[0]} linhas × {dados_treino.shape[1]} colunas"))
    except Exception as e:
        mensagens.append(("error", f"Erro ao carregar treino: {e}"))
if arquivo_teste:
    try:
        dados_teste = carregar_csv(arquivo_teste)
        dados_teste = normalizar_colunas(dados_teste)
        mensagens.append(("success", f"Teste carregado: {dados_teste.shape[0]} linhas × {dados_teste.shape[1]} colunas"))
    except Exception as e:
        mensagens.append(("error", f"Erro ao carregar teste: {e}"))

for tipo, texto in mensagens:
    getattr(st, tipo)(texto)

if dados_treino is None:
    st.warning("⏳ Carregue o arquivo de treino para prosseguir.")
    st.stop()

# ---------------- EDA simples com diagnóstico ----------------
st.header("📊 EDA Rápida & Diagnóstico")
st.write("Colunas do dataset (exemplo):", list(dados_treino.columns[:40]))
st.write("Resumo de tipos:")
st.write(dados_treino.dtypes.value_counts().to_frame("count"))

# coerção de numericos esperados (opcional: garante que sensores sejam numéricos)
cols_coerce = ["temperatura_ar","temperatura_processo","umidade_relativa","velocidade_rotacional","torque","desgaste_da_ferramenta"] + [COL_ALVO_BINARIA] + COLS_FALHA
dados_treino, rel_coerce = coagir_e_reportar_numericos(dados_treino, cols_coerce)
if any(v>0 for v in rel_coerce.values()):
    st.info(f"Coerção numérica feita em colunas (novos NaNs detectados): { {k:v for k,v in rel_coerce.items() if v>0} }")

# mostrar alguns stats simples
st.subheader("Valores ausentes (%) — top 20")
st.write((dados_treino.isna().mean().sort_values(ascending=False) * 100).head(20))

# ---------------- Engenharia e preparação ----------------
st.header("🔧 Preparação & Engenharia")
dados_treino_eng = engenhar_features(dados_treino, usar_dif_temp, usar_potencia, usar_eficiencia)

# Encoding básico da coluna 'tipo' se existir
le_tipo = None
if "tipo" in dados_treino_eng.columns:
    le_tipo = LabelEncoder()
    dados_treino_eng["tipo_encoded"] = le_tipo.fit_transform(dados_treino_eng["tipo"].fillna("missing"))

# montar lista de features (remover ids e alvos)
excluir = set(["id","id_produto","tipo", COL_ALVO_BINARIA] + COLS_FALHA)
features = [c for c in dados_treino_eng.columns if c not in excluir]
if len(features) == 0:
    st.error("❌ Nenhuma feature válida encontrada após exclusões. Verifique o dataset.")
    st.stop()
st.success(f"{len(features)} features selecionadas para treino.")

# normalização opcional
scaler = None
if normalizar:
    scaler = StandardScaler()

# preparar targets
targets = preparar_targets(dados_treino_eng)

# ---------------- Seleção do tipo de target com diagnóstico robusto ----------------
st.header("🤖 Definindo alvo (target)")

tipo_target = None
y = None
le_target = None

if tipo_modelagem == "Multirrótulo":
    if "multirrotulo" not in targets:
        st.error("❌ Dados não contêm colunas de falha necessárias (fdf, fdc, fp, fte, fa) para multirrótulo.")
        st.stop()
    y = targets["multirrotulo"]
    cols_falha = targets["colunas_multirrotulo"]
    tipo_target = "multirrotulo"

elif tipo_modelagem == "Binária":
    if "binario" not in targets:
        st.error("❌ Dados não contêm informações de falha necessárias para modelagem binária (campo 'falha_maquina').")
        st.stop()
    y = targets["binario"]
    if pd.Series(y).nunique() < 2:
        st.error("❌ Alvo binário tem somente uma classe. Impossível treinar.")
        st.stop()
    tipo_target = "binario"

else:  # Multiclasse
    if "multiclasse" not in targets:
        st.error("❌ Dados não contêm informações de falha necessárias para modelagem multiclasse.")
        st.stop()
    y = targets["multiclasse"]
    if len(set(y)) < 2:
        st.error("❌ Alvo multiclasse tem somente uma classe. Impossível treinar.")
        st.stop()
    le_target = LabelEncoder()
    y = le_target.fit_transform(pd.Series(y).fillna("sem_valor"))
    tipo_target = "multiclasse"

# ---------------- construir X (e transformar categóricas) ----------------
X_raw = dados_treino_eng[features].copy()

# converter X para matriz numérica (get_dummies + imputação)
X_proc, colunas_final = preparar_X_dataframe(X_raw, referencia_cols=None)

# aplicar scaler se necessário
if scaler is not None and not X_proc.empty:
    X_proc[colunas_final] = scaler.fit_transform(X_proc[colunas_final])

st.write(f"Dimensão X (após processamento): {X_proc.shape}")

# ---------------- split treino/validação ----------------
if tipo_target == "multirrotulo":
    # y é um array (n, n_classes)
    X_tr, X_va, y_tr, y_va = train_test_split(X_proc, y, test_size=percentual_teste/100, random_state=semente)
else:
    X_tr, X_va, y_tr, y_va = train_test_split(X_proc, y, test_size=percentual_teste/100, random_state=semente, stratify=y if len(np.unique(y))>1 else None)

# ---------------- TRY FIT com diagnóstico (captura falhas de dtype / NaNs) ----------------
st.header("🔄 Treinamento (com diagnóstico)")
try:
    # garantir y em dtype correto
    if tipo_target == "multirrotulo":
        y_tr = np.array(y_tr).astype(int)
    else:
        # se y_tr for object, encoder já aplicado para multiclasse; para binário garantimos int
        y_tr = np.array(y_tr).astype(int)

    # Treinar
    if tipo_target == "multirrotulo":
        base = RandomForestClassifier(n_estimators=n_estimators, random_state=semente, n_jobs=-1)
        modelo = MultiOutputClassifier(base)
        modelo.fit(X_tr, y_tr)
        pred_va = modelo.predict(X_va)

        # métricas por classe
        metricas = {col: accuracy_score(y_va[:,i], pred_va[:,i]) for i,col in enumerate(cols_falha)}
        st.subheader("📈 Métricas por classe (validação)")
        cols_display = st.columns(len(cols_falha))
        for i,c in enumerate(cols_falha):
            with cols_display[i]:
                st.metric(c, f"{metricas[c]:.3f}")
        st.metric("Média (macro)", f"{np.mean(list(metricas.values())):.3f}")

    else:
        modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=semente, n_jobs=-1)
        modelo.fit(X_tr, y_tr)
        pred_va = modelo.predict(X_va)

        acc = accuracy_score(y_va, pred_va)
        st.metric("Acurácia (validação)", f"{acc:.4f}")

        st.subheader("📋 Relatório (validação)")
        st.dataframe(classification_report_df(y_va, pred_va), use_container_width=True)

        st.subheader("🔎 Matriz de Confusão")
        st.plotly_chart(matriz_confusao_fig(y_va, pred_va), use_container_width=True)

except Exception as e:
    # diagnosticar colunas problemáticas
    st.error("❌ Erro durante o treinamento do modelo.")
    st.error(str(e))
    st.write("Diagnóstico rápido — verifique os itens abaixo:")
    st.write("- Dimensões X_tr:", getattr(X_tr, "shape", "n/a"))
    st.write("- Tipos em X_tr (preview):")
    try:
        dtypes = X_tr.dtypes.to_dict()
        # mostrar colunas que não são numéricas
        nao_num = [c for c,t in dtypes.items() if not np.issubdtype(t, np.number)]
        st.write("Colunas não-numéricas (se houver):", nao_num)
        st.write(pd.Series({c:str(t) for c,t in dtypes.items()}).head(50))
    except Exception:
        st.write("Não foi possível listar dtypes.")
    st.write("Valores nulos por coluna (top 10):")
    try:
        st.write(X_tr.isna().sum().sort_values(ascending=False).head(10))
    except Exception:
        pass
    st.stop()

# ---------------- Predição no teste e CSV para API ----------------
if dados_teste is not None:
    st.header("🎯 Predições no conjunto de teste")
    dados_teste_eng = engenhar_features(dados_teste, usar_dif_temp, usar_potencia, usar_eficiencia)
    if "tipo" in dados_teste_eng.columns and le_tipo is not None:
        dados_teste_eng["tipo_encoded"] = le_tipo.transform(dados_teste_eng["tipo"].fillna("missing"))

    # selecionar mesmas features
    X_te_raw = dados_teste_eng[features].copy()
    X_te_proc, _ = preparar_X_dataframe(X_te_raw, referencia_cols=colunas_final)

    if scaler is not None and not X_te_proc.empty:
        X_te_proc[colunas_final] = scaler.transform(X_te_proc[colunas_final])

    # gerar df de saída com colunas fdf,fdc,fp,fte,fa inicializado com zeros
    df_pred_api = pd.DataFrame(0, index=X_te_proc.index, columns=[c.upper() for c in COLS_FALHA], dtype=int)

    # pred
    if tipo_target == "multirrotulo":
        if usar_proba and hasattr(modelo, "estimators_"):
            # tenta usar probabilidades por estimador se houver
            probs = []
            for est in modelo.estimators_:
                if hasattr(est, "predict_proba"):
                    p = est.predict_proba(X_te_proc)
                    probs.append(p[:,1] if p.shape[1] > 1 else p[:,0])
                else:
                    probs.append(np.zeros(X_te_proc.shape[0]))
            P = np.vstack(probs).T
            mask = (P >= threshold_api)
            df_pred_api[:] = mask.astype(int)
        else:
            pred_te = modelo.predict(X_te_proc)
            df_pred_api[:] = pred_te.astype(int)

    elif tipo_target == "multiclasse":
        pred_te = modelo.predict(X_te_proc)
        classes_nom = le_target.inverse_transform(pred_te)
        for i,c in enumerate(classes_nom):
            if c in [col.lower() for col in COLS_FALHA]:
                df_pred_api.iloc[i, [ [col.lower() for col in COLS_FALHA].index(c) ]] = 1
    else:  # binário
        if usar_proba and hasattr(modelo, "predict_proba"):
            p1 = modelo.predict_proba(X_te_proc)[:,1]
            y_bin = (p1 >= threshold_api).astype(int)
        else:
            y_bin = modelo.predict(X_te_proc).astype(int)
        # escolher fallback: a falha mais frequente no treino (se existir multilabel no treino)
        if all(c in dados_treino_eng.columns for c in COLS_FALHA):
            freq = dados_treino_eng[COLS_FALHA].sum().sort_values(ascending=False)
            classe_mais_comum = freq.index[0].upper()
        else:
            classe_mais_comum = "FDF"
        df_pred_api.loc[y_bin==1, classe_mais_comum] = 1

    st.subheader("🎯 Estatísticas das predições")
    st.write(df_pred_api.sum())
    st.download_button("📥 Baixar CSV (API)", df_pred_api.to_csv(index=False).encode('utf-8'), "bootcamp_predictions.csv", "text/csv")

st.markdown("---")
st.markdown("Desenvolvido manualmente — versão com correções de schema e pré-processamento.")
