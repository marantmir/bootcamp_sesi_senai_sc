"""
Funções auxiliares para o projeto do bootcamp
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

# Lista das falhas que queremos prever
FALHAS = ["FDF", "FDC", "FP", "FTE", "FA"]

# Colunas que vamos usar como entrada pro modelo
COLUNAS_USAR = [
    "tipo", 
    "temperatura_ar", 
    "temperatura_processo", 
    "umidade_relativa",
    "velocidade_rotacional", 
    "torque", 
    "desgaste_ferramenta"
]

def carregar_dados(caminho):
    """
    Tenta carregar um CSV. 
    Às vezes dá erro com encoding, então tentei deixar mais robusto
    """
    try:
        print(f"Tentando abrir {caminho}...")
        df = pd.read_csv(caminho)
        print(f"Deu certo! {len(df)} linhas carregadas.")
        return df
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {caminho}")
        return None
    except Exception as e:
        print(f"Deu ruim ao abrir {caminho}: {e}")
        # Tenta de outro jeito às vezes funciona
        try:
            df = pd.read_csv(caminho, encoding='latin-1')
            print("Consegui abrir com encoding latin-1!")
            return df
        except:
            print("Não deu certo mesmo...")
            return None

def limpar_dados(df):
    """
    Faz uma limpeza básica nos dados.
    """
    if df is None:
        print("Nada para limpar - dados vazios")
        return None
    
    df_copy = df.copy()
    
    # Tira duplicados se tiver coluna id
    if "id" in df_copy.columns:
        antes = len(df_copy)
        df_copy = df_copy.drop_duplicates(subset=["id"])
        depois = len(df_copy)
        if antes != depois:
            print(f"Removidas {antes-depois} linhas duplicadas")
    
    # Aqueles valores '?' que dão problema
    df_copy.replace("?", np.nan, inplace=True)
    
    # Converte as colunas numéricas - às vezes vem como string
    colunas_numericas = [col for col in COLUNAS_USAR if col != 'tipo']
    
    for coluna in colunas_numericas:
        if coluna in df_copy.columns:
            # Tenta converter, se der erro coloca NaN
            df_copy[coluna] = pd.to_numeric(df_copy[coluna], errors="coerce")
            # Conta quantos NaN tem agora
            nans = df_copy[coluna].isna().sum()
            if nans > 0:
                print(f"Cuidado: {nans} valores NaN na coluna {coluna}")
    
    return df_copy

def preparar_preprocessamento(dados):
    """
    Prepara o preprocessamento dos dados.
    """
    # Colunas categóricas (só o tipo mesmo)
    colunas_cat = ["tipo"] if "tipo" in dados.columns else []
    
    # Colunas numéricas
    colunas_num = [col for col in COLUNAS_USAR if col != 'tipo' and col in dados.columns]
    
    if not colunas_num:
        print("Não encontrei colunas numéricas!")
        return None, [], []
    
    # Pipeline para números - aprendi que é bom padronizar
    pipe_numerico = Pipeline([
        ("preencher", SimpleImputer(strategy="median")),  # median é mais robusto
        ("escalar", StandardScaler())  # importante para muitos algoritmos
    ])

    # Pipeline para categorias
    pipe_categorico = Pipeline([
        ("preencher", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Junta tudo - isso é bem legal do sklearn
    preprocessador = ColumnTransformer([
        ("numerico", pipe_numerico, colunas_num),
        ("categorico", pipe_categorico, colunas_cat)
    ])
    
    return preprocessador, colunas_num, colunas_cat

def separar_dados_treino_teste(dados, tamanho_teste=0.2, random_state=42):
    """
    Separa os dados em treino e teste.
    """
    if dados is None:
        return None, None, None, None
        
    df = dados.copy()
    
    # Colunas que não são features
    colunas_para_tirar = FALHAS + ["id", "id_produto", "falha_maquina"]
    colunas_para_tirar = [col for col in colunas_para_tirar if col in df.columns]
    
    # Features (X)
    X = df.drop(columns=colunas_para_tirar)
    
    # Targets (y) - só se tiver as colunas de falha
    if all(falha in df.columns for falha in FALHAS):
        y = df[FALHAS]
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y, test_size=tamanho_teste, random_state=random_state
        )
        print(f"Separado: {len(X_treino)} treino, {len(X_teste)} teste")
        return X_treino, X_teste, y_treino, y_teste
    else:
        print("Arquivo não tem as colunas de falha - deve ser de teste")
        return X, None, None, None

def salvar_modelo_treinado(modelo, caminho):
    """
    Salva o modelo treinado.
    """
    try:
        # Cria a pasta se não existir
        pasta = os.path.dirname(caminho)
        if pasta and not os.path.exists(pasta):
            os.makedirs(pasta)
            print(f"Criada pasta {pasta}")
        
        joblib.dump(modelo, caminho)
        print(f"Modelo salvo em {caminho}")
        return True
    except Exception as e:
        print(f"Erro ao salvar modelo: {e}")
        return False

def carregar_modelo_treinado(caminho):
    """
    Carrega um modelo salvo.
    """
    try:
        if not os.path.exists(caminho):
            print(f"Arquivo não existe: {caminho}")
            return None
        
        modelo = joblib.load(caminho)
        print(f"Modelo carregado de {caminho}")
        return modelo
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None