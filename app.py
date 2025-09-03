"""
Sistema Inteligente de Manutenção Preditiva
Autor: Marco (código estruturado manualmente, com foco em clareza e robustez)

Decisões de projeto:
- Somente Plotly para visualizações (menos dependências).
- Validações de schema rígidas + mensagens úteis.
- Três estratégias de modelagem: Binária, Multiclasse e Multirrótulo.
- Engenharia de atributos mínima porém útil (dif. de temperatura, potência, eficiência).
- Métricas e visuais interpretáveis no Streamlit.
- Geração do CSV no formato da API (FDF, FDC, FP, FTE, FA).

Referências do desafio: ver documentos no repositório.
"""

import io
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.multioutput import MultiOutputClassifier

import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------------------------------------------
# Configuração da página
# -------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Sistema de Manutenção Preditiva",
    page_icon="🔧",
    layout="wide"
)
st.title("🔧 Sistema Inteligente de Manutenção Preditiva")
st.caption(
    "Projeto alinhado ao desafio do Bootcamp CDIA — "
    "treino no *train*, predição no *test* e geração de CSV para a API. "
    "Documentação de campos e API consultadas nos PDFs do desafio."
)

# -------------------------------------------------------------------------------------------------
# Constantes do desafio (conforme documentação do projeto)
# -------------------------------------------------------------------------------------------------
COLS_BASE = [
    "id", "id_produto", "tipo", "temperatura_ar", "temperatura_processo",
    "umidade_relativa", "velocidade_rotacional", "torque", "desgaste_da_ferramenta"
]
# Alvos
COLS_FALHA = ["FDF", "FDC", "FP", "FTE", "FA"]
COL_ALVO_BINARIA = "falha_maquina"

# -------------------------------------------------------------------------------------------------
# Utilitários
# -------------------------------------------------------------------------------------------------
def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes de colunas para minúsculo e underscore simples."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("ã", "a")
        .str.replace("á", "a")
        .str.replace("â", "a")
        .str.replace("é", "e")
        .str.replace("ê", "e")
        .str.replace("í", "i")
        .str.replace("ó", "o")
        .str.replace("ô", "o")
        .str.replace("ú", "u")
        .str.replace("ç", "c")
    )
    return df


def validar_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Confere presença das colunas essenciais do desafio e retorna faltantes por grupo."""
    faltantes_base = [c for c in [c.lower() for c in COLS_BASE] if c not in df.columns]
    faltantes_multilabel = [c.lower() for c in COLS_FALHA if c.lower() not in df.columns]
    faltante_binaria = [] if COL_ALVO_BINARIA in df.columns else [COL_ALVO_BINARIA]
    return {
        "base": faltantes_base,
        "multilabel": faltantes_multilabel,
        "binaria": faltante_binaria,
    }


def coagir_numericos(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Converte colunas para numérico (coerce), reportando quantidade de não-conversões por coluna.
    Isso evita erros como 'Could not convert string to numeric' ao calcular médias.
    """
    df = df.copy()
    relatorio = {}
    for c in cols:
        if c in df.columns:
            antes_na = df[c].isna().sum()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            depois_na = df[c].isna().sum()
            relatorio[c] = max(0, depois_na - antes_na)
    return df, relatorio


def engenhar_features(
    df: pd.DataFrame,
    usar_dif_temp: bool = True,
    usar_potencia: bool = True,
    usar_eficiencia: bool = True
) -> pd.DataFrame:
    """Cria atributos adicionais simples e interpretáveis."""
    df = df.copy()
    if usar_dif_temp and {"temperatura_processo", "temperatura_ar"}.issubset(df.columns):
        df["dif_temperatura"] = df["temperatura_processo"] - df["temperatura_ar"]
        # Evita divisão por zero com epsilon
        df["razao_temperatura"] = df["temperatura_processo"] / (df["temperatura_ar"] + 1e-6)

    if usar_potencia and {"torque", "velocidade_rotacional"}.issubset(df.columns):
        # Potência ~ torque * velocidade (escala kW ~ /1000 para reduzir ordem de grandeza)
        df["potencia_kw"] = (df["torque"] * df["velocidade_rotacional"]) / 1000.0

    if usar_eficiencia and {"torque", "desgaste_da_ferramenta"}.issubset(df.columns):
        df["eficiencia_ferramenta"] = df["torque"] / (df["desgaste_da_ferramenta"] + 1.0)

    return df


def preparar_targets(df: pd.DataFrame) -> Dict[str, object]:
    """
    Prepara dicionário com possíveis alvos:
      - binario: 0/1 (usa 'falha_maquina' se existir, caso contrário max das 5 classes)
      - multiclasse: Sem_Falha | FDF|FDC|FP|FTE|FA | Multiplas_Falhas
      - multirrotulo: matriz (n_amostras, 5) com colunas em COLS_FALHA
    """
    targets = {}

    tem_multilabel = all(c in df.columns for c in COLS_FALHA)
    tem_binario = COL_ALVO_BINARIA in df.columns

    if tem_multilabel:
        y_ml = df[COLS_FALHA].astype(int).values
        targets["multirrotulo"] = y_ml
        targets["colunas_multirrotulo"] = COLS_FALHA

        # Binário preferencialmente do próprio campo; senão, derivado
        if tem_binario:
            y_bin = pd.to_numeric(df[COL_ALVO_BINARIA], errors="coerce").fillna(0).astype(int)
        else:
            y_bin = df[COLS_FALHA].max(axis=1).astype(int)
        targets["binario"] = y_bin

        # Multiclasse
        def tipo_principal(row):
            ativos = [c for c in COLS_FALHA if row[c] == 1]
            if len(ativos) == 0:
                return "Sem_Falha"
            if len(ativos) == 1:
                return ativos[0]
            return "Multiplas_Falhas"

        targets["multiclasse"] = df[COLS_FALHA].apply(tipo_principal, axis=1)

    else:
        # Sem multilabel, ainda podemos tentar binário se existir
        if tem_binario:
            y_bin = pd.to_numeric(df[COL_ALVO_BINARIA], errors="coerce").fillna(0).astype(int)
            targets["binario"] = y_bin

    return targets


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).transpose()


def matriz_confusao_fig(y_true, y_pred, titulo: str = "Matriz de Confusão"):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm, text_auto=True, aspect="auto", title=titulo, color_continuous_scale="Blues"
    )
    fig.update_xaxes(title="Predito")
    fig.update_yaxes(title="Real")
    return fig


# -------------------------------------------------------------------------------------------------
# Sidebar — Configurações
# -------------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configurações")

    arquivo_treino = st.file_uploader("📁 Selecione o Bootcamp_train.csv", type=["csv"])
    arquivo_teste = st.file_uploader("📁 Selecione o Bootcamp_test.csv (opcional)", type=["csv"])

    st.subheader("🤖 Modelagem")
    tipo_modelagem = st.selectbox(
        "Tipo de Modelagem",
        ["Multirrótulo (cada tipo de falha)", "Binária (qualquer falha)", "Multiclasse (tipo principal)"]
    )
    percentual_teste = st.slider("% Validação", 10, 40, 20, step=5)
    semente = st.slider("Semente Aleatória", 0, 999, 42)
    n_estimators = st.slider("Árvores (RandomForest)", 100, 600, 300, step=50)

    st.subheader("🔧 Engenharia de Features")
    usar_dif_temp = st.checkbox("Diferença/razão de temperatura", value=True)
    usar_potencia = st.checkbox("Potência (torque × velocidade) [kW]", value=True)
    usar_eficiencia = st.checkbox("Eficiência da ferramenta", value=True)
    normalizar = st.checkbox("Normalizar features numéricas", value=False)

    st.subheader("🧪 Geração do CSV (API)")
    usar_proba = st.checkbox("Usar probabilidades (quando disponíveis)", value=True)
    threshold_api = st.slider("Threshold para API (0.0-1.0)", 0.0, 1.0, 0.5, 0.05)
    st.caption("A API espera colunas FDF, FDC, FP, FTE, FA. Vide documentação oficial.")
    # API oficial conforme instruções do desafio
    st.caption("Endpoints: /users/register e /evaluate/multilabel_metrics.")

# -------------------------------------------------------------------------------------------------
# Carregamento e validações iniciais
# -------------------------------------------------------------------------------------------------
@st.cache_data
def carregar_csv(f) -> pd.DataFrame:
    return pd.read_csv(f)

dados_treino, dados_teste = None, None
mensagens = []

if arquivo_treino is not None:
    try:
        dados_treino = carregar_csv(arquivo_treino)
        dados_treino = normalizar_colunas(dados_treino)
        mensagens.append(("success", f"Treino: {dados_treino.shape[0]} linhas × {dados_treino.shape[1]} colunas"))
    except Exception as e:
        mensagens.append(("error", f"Erro ao carregar treino: {e}"))

if arquivo_teste is not None:
    try:
        dados_teste = carregar_csv(arquivo_teste)
        dados_teste = normalizar_colunas(dados_teste)
        mensagens.append(("success", f"Teste: {dados_teste.shape[0]} linhas × {dados_teste.shape[1]} colunas"))
    except Exception as e:
        mensagens.append(("error", f"Erro ao carregar teste: {e}"))

for tipo, txt in mensagens:
    getattr(st, tipo)(txt)

if dados_treino is None:
    st.warning("Carregue o arquivo de treino para iniciar.")
    st.stop()

# -------------------------------------------------------------------------------------------------
# Análise Exploratória (EDA)
# -------------------------------------------------------------------------------------------------
st.header("📊 Análise Exploratória")

# 1) Checagem de schema conforme docs do desafio
checagem = validar_schema(dados_treino)
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Amostras", f"{dados_treino.shape[0]:,}")
with col_b:
    st.metric("Colunas", dados_treino.shape[1])
with col_c:
    st.metric("Faltam (base)", len(checagem["base"]))
with col_d:
    st.metric("Faltam (multirrótulo)", len(checagem["multilabel"]))

if any(checagem.values()):
    with st.expander("Detalhes da checagem de schema"):
        st.write(checagem)
        st.caption("As colunas esperadas seguem a especificação do desafio (train/test, alvos e atributos).")

# 2) Tipagem e coerção numérica (evita erro ao calcular médias)
cols_numericas_esperadas = [
    "temperatura_ar", "temperatura_processo", "umidade_relativa",
    "velocidade_rotacional", "torque", "desgaste_da_ferramenta"
] + [c for c in [COL_ALVO_BINARIA] if c in dados_treino.columns] + [c for c in COLS_FALHA if c in dados_treino.columns]

dados_treino, rel_num = coagir_numericos(dados_treino, cols_numericas_esperadas)
if any(v > 0 for v in rel_num.values()):
    st.info(f"Coerção numérica aplicada: { {k: v for k, v in rel_num.items() if v > 0} }")

# 3) Resumo de ausentes
ausentes_pct = dados_treino.isna().mean().sort_values(ascending=False) * 100
fig_aus = px.bar(
    ausentes_pct.head(20),
    title="Top 20 colunas com mais ausentes (%)",
    labels={"value": "% Ausentes", "index": "Coluna"}
)
st.plotly_chart(fig_aus, use_container_width=True)

# 4) Distribuições de alvo (se houver)
col1, col2 = st.columns(2)
with col1:
    if COL_ALVO_BINARIA in dados_treino.columns:
        taxa_falha = pd.to_numeric(dados_treino[COL_ALVO_BINARIA], errors="coerce").mean()
        st.metric("Taxa de Falha (binário)", f"{(0 if pd.isna(taxa_falha) else taxa_falha):.1%}")

with col2:
    if all(c in dados_treino.columns for c in COLS_FALHA):
        contagens = dados_treino[COLS_FALHA].sum().sort_values(ascending=True)
        fig = px.bar(
            x=contagens.values,
            y=contagens.index,
            orientation="h",
            title="Distribuição por tipo de falha (multirrótulo)",
            labels={"x": "Quantidade", "y": "Tipo de falha"},
            color=contagens.values,
            color_continuous_scale="Reds"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# 5) Correlação de features numéricas
num_cols = dados_treino.select_dtypes(include=[np.number]).columns
# Exclui alvos da correlação de features
cols_feat = [c for c in num_cols if c not in set([COL_ALVO_BINARIA] + COLS_FALHA)]
if len(cols_feat) >= 2:
    corr = dados_treino[cols_feat].corr()
    fig_corr = px.imshow(corr, title="Matriz de correlação (features)", color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# Preparação dos dados
# -------------------------------------------------------------------------------------------------
st.header("🔧 Preparação dos Dados")

# Engenharia de atributos (opcional)
dados_treino_eng = engenhar_features(
    dados_treino,
    usar_dif_temp=usar_dif_temp,
    usar_potencia=usar_potencia,
    usar_eficiencia=usar_eficiencia
)

# Encoding de 'tipo' (L/M/H) se existir
le_tipo: Optional[LabelEncoder] = None
if "tipo" in dados_treino_eng.columns:
    le_tipo = LabelEncoder()
    dados_treino_eng["tipo_encoded"] = le_tipo.fit_transform(dados_treino_eng["tipo"])

# Seleção de features (exclui IDs, alvos)
excluir = set(["id", "id_produto", "tipo", COL_ALVO_BINARIA] + COLS_FALHA)
features = [c for c in dados_treino_eng.columns if c not in excluir]
X = dados_treino_eng[features].copy()

# Normalização (se escolhida)
scaler: Optional[StandardScaler] = None
if normalizar and not X.empty:
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

st.success(f"Features prontas: {len(features)} colunas | {X.shape[0]} amostras")

# Targets possíveis a partir dos dados
targets = preparar_targets(dados_treino_eng)

# -------------------------------------------------------------------------------------------------
# Treinamento
# -------------------------------------------------------------------------------------------------
st.header("🤖 Treinamento do Modelo")

tipo_target = None
y = None
le_target: Optional[LabelEncoder] = None

if tipo_modelagem.startswith("Multirrótulo"):
    if "multirrotulo" in targets:
        y = targets["multirrotulo"]
        cols_falha = targets["colunas_multirrotulo"]
        tipo_target = "multirrotulo"
    else:
        st.error("❌ Dados não contêm colunas de falha necessárias (FDF, FDC, FP, FTE, FA) para multirrótulo.")
        st.stop()

elif tipo_modelagem.startswith("Binária"):
    if "binario" in targets:
        y = targets["binario"]
        if pd.Series(y).nunique() < 2:
            st.error("❌ Alvo binário não possui ao menos duas classes.")
            st.stop()
        tipo_target = "binario"
    else:
        st.error("❌ Dados não contêm informações de falha necessárias para modelagem binária.")
        st.stop()

else:  # Multiclasse
    if "multiclasse" in targets:
        y = targets["multiclasse"]
        if len(set(y)) < 2:
            st.error("❌ Alvo multiclasse não possui ao menos duas classes.")
            st.stop()
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)  # transforma rótulos para 0..K-1
        tipo_target = "multiclasse"
    else:
        st.error("❌ Dados não contêm informações de falha necessárias para modelagem multiclasse.")
        st.stop()

# Split
if tipo_target == "multirrotulo":
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=percentual_teste/100, random_state=semente)
else:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=percentual_teste/100, random_state=semente,
        stratify=y if len(np.unique(y)) > 1 else None
    )

# Treino
with st.spinner("Treinando modelo..."):
    if tipo_target == "multirrotulo":
        base = RandomForestClassifier(n_estimators=n_estimators, random_state=semente, n_jobs=-1)
        modelo = MultiOutputClassifier(base)
        modelo.fit(X_tr, y_tr)
        pred_va = modelo.predict(X_va)

        # Métricas por classe (acurácia simples)
        metricas = {}
        for i, classe in enumerate(cols_falha):
            metricas[classe] = accuracy_score(y_va[:, i], pred_va[:, i])

        st.subheader("📈 Acurácia por classe (validação)")
        cols = st.columns(len(cols_falha))
        for i, c in enumerate(cols_falha):
            with cols[i]:
                st.metric(c, f"{metricas[c]:.3f}")
        st.metric("Média (macro)", f"{np.mean(list(metricas.values())):.3f}")

        # Importância média das features (floresta por saída)
        try:
            importancias = np.mean([est.feature_importances_ for est in modelo.estimators_], axis=0)
            df_imp = pd.DataFrame({"feature": features, "importance": importancias}).sort_values("importance", ascending=True)
            fig_imp = px.bar(df_imp.tail(12), x="importance", y="feature", orientation="h", title="Top features (média)")
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            st.info("Importâncias não disponíveis para este estimador.")

    else:
        modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=semente, n_jobs=-1)
        modelo.fit(X_tr, y_tr)
        pred_va = modelo.predict(X_va)

        acc = accuracy_score(y_va, pred_va)
        st.metric("Acurácia (validação)", f"{acc:.4f}")

        # Relatório detalhado
        st.subheader("📋 Relatório (validação)")
        df_rep = classification_report_df(y_va, pred_va)
        st.dataframe(df_rep, use_container_width=True)

        # Matriz de confusão
        st.subheader("🔎 Matriz de Confusão")
        fig_cm = matriz_confusao_fig(y_va, pred_va)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Importância das features
        if hasattr(modelo, "feature_importances_"):
            df_imp = pd.DataFrame({"feature": features, "importance": modelo.feature_importances_})
            df_imp = df_imp.sort_values("importance", ascending=True)
            fig_imp = px.bar(df_imp.tail(12), x="importance", y="feature", orientation="h", title="Top features")
            st.plotly_chart(fig_imp, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# Predição no conjunto de teste + CSV para API
# -------------------------------------------------------------------------------------------------
if dados_teste is not None:
    st.header("🎯 Predições no Conjunto de Teste")

    # Repete pipeline no teste
    dados_teste_eng = engenhar_features(
        dados_teste,
        usar_dif_temp=usar_dif_temp,
        usar_potencia=usar_potencia,
        usar_eficiencia=usar_eficiencia
    )
    if "tipo" in dados_teste_eng.columns and le_tipo is not None:
        dados_teste_eng["tipo_encoded"] = le_tipo.transform(dados_teste_eng["tipo"])

    feat_teste = [c for c in features if c in dados_teste_eng.columns]
    X_te = dados_teste_eng[feat_teste].copy()
    if scaler is not None and not X_te.empty:
        X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)

    # Predições
    with st.spinner("Gerando predições..."):
        # Saída final obrigatória para API: colunas FDF, FDC, FP, FTE, FA (0/1)
        df_pred_api = pd.DataFrame(0, index=X_te.index, columns=COLS_FALHA, dtype=int)

        if tipo_target == "multirrotulo":
            # MultiOutputClassifier -> predições 0/1 por classe
            if usar_proba and hasattr(modelo, "predict_proba"):
                # Nem todo base estimator do MultiOutput expõe proba corretamente.
                # Quando disponível, permite threshold custom.
                probas = []
                for est in modelo.estimators_:
                    if hasattr(est, "predict_proba"):
                        p = est.predict_proba(X_te)
                        # Proba da classe positiva é [:,1] se binário
                        probas.append(p[:, 1] if p.shape[1] > 1 else p[:, 0])
                    else:
                        probas.append(np.zeros(X_te.shape[0]))
                P = np.vstack(probas).T  # (n_amostras, 5)
                df_pred_api = pd.DataFrame((P >= threshold_api).astype(int), index=X_te.index, columns=COLS_FALHA)
            else:
                pred_te = modelo.predict(X_te)
                df_pred_api = pd.DataFrame(pred_te, index=X_te.index, columns=COLS_FALHA)

        elif tipo_target == "multiclasse":
            pred_te = modelo.predict(X_te)
            # Converte classe nominal para one-hot das 5 falhas
            # Multiplas_Falhas vira todas 0 (opção conservadora) — comentado explicitamente:
            # como não sabemos distribuição interna, evitamos chutar várias falhas.
            classes_nominais = le_target.inverse_transform(pred_te)
            for i, c in enumerate(classes_nominais):
                if c in COLS_FALHA:
                    df_pred_api.loc[df_pred_api.index[i], c] = 1

        else:  # Binário
            # Para binário, se houver probabilidade, aplicamos threshold e, quando 1,
            # atribuímos à classe de falha mais frequente no treino (estratégia simples e reprodutível).
            if usar_proba and hasattr(modelo, "predict_proba"):
                p1 = modelo.predict_proba(X_te)[:, 1]
                y_bin = (p1 >= threshold_api).astype(int)
            else:
                y_bin = modelo.predict(X_te)

            if all(c in dados_treino_eng.columns for c in COLS_FALHA):
                # Classe de falha mais frequente entre as positivas no treino
                dist = dados_treino_eng[COLS_FALHA].sum().sort_values(ascending=False)
                classe_mais_comum = dist.index[0]
            else:
                classe_mais_comum = "FDF"  # fallback explícito

            df_pred_api.loc[y_bin == 1, classe_mais_comum] = 1

    # Painel de estatísticas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de amostras", len(df_pred_api))
    with col2:
        st.metric("Total de falhas (preditas)", int(df_pred_api.sum().sum()))
    with col3:
        st.metric("Taxa com alguma falha", f"{(df_pred_api.sum(axis=1) > 0).mean():.1%}")

    contagens_pred = df_pred_api.sum().sort_values(ascending=True)
    fig_pred = px.bar(
        x=contagens_pred.values,
        y=contagens_pred.index,
        orientation="h",
        title="Distribuição das predições por tipo de falha",
        labels={"x": "Qtd", "y": "Tipo de falha"}
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.subheader("👀 Amostra do CSV gerado (para API)")
    st.dataframe(df_pred_api.head(20), use_container_width=True)

    # CSV para download
    csv_bytes = df_pred_api.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Baixar CSV para API",
        data=csv_bytes,
        file_name="bootcamp_predictions.csv",
        mime="text/csv"
    )

    with st.expander("🌐 (Opcional) Testar na API do desafio — usar com cautela"):
        st.caption(
            "A disponibilidade da API não é garantida. Use somente para avaliação final, "
            "após validar localmente. Veja PDFs de instruções oficiais."
        )
        email_api = st.text_input("Email (cadastro)", "")
        senha_api = st.text_input("Senha", type="password", value="")
        th = st.slider("Threshold para /evaluate", 0.0, 1.0, float(threshold_api), 0.05)

        if st.button("Obter token e avaliar"):
            if not email_api or not senha_api:
                st.warning("Informe email e senha.")
            else:
                try:
                    # Cadastro (ou obtenção de token)
                    r = requests.post(
                        "http://34.193.187.218:5000/users/register",
                        json={"email": email_api, "password": senha_api},
                        timeout=20
                    )
                    if r.status_code != 200:
                        st.error(f"Erro no registro: {r.status_code} | {r.text}")
                    else:
                        token = r.json().get("token", None)
                        if not token:
                            st.error("Token não retornado pela API.")
                        else:
                            st.success("Token obtido!")

                            # Avaliação
                            headers = {"X-API-Key": token}
                            files = {"file": ("submission.csv", csv_bytes, "text/csv")}
                            params = {"threshold": th}
                            r2 = requests.post(
                                "http://34.193.187.218:5000/evaluate/multilabel_metrics",
                                headers=headers, files=files, params=params, timeout=40
                            )
                            st.write(f"Status: {r2.status_code}")
                            if r2.status_code == 200:
                                try:
                                    st.json(r2.json())
                                except Exception:
                                    st.text(r2.text)
                            else:
                                st.error(r2.text)
                except Exception as e:
                    st.error(f"Falha de comunicação com a API: {e}")

# -------------------------------------------------------------------------------------------------
# Rodapé
# -------------------------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "Desenvolvido manualmente para o Bootcamp de Ciência de Dados e IA — "
    "com foco em rastreabilidade das decisões, clareza e robustez do pipeline."
)
