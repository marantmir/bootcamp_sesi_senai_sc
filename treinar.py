"""
Script para treinar o modelo de previsão de falhas
Fiz baseado no que aprendi no curso e alguns exemplos online
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import time
from utilitarios import (
    carregar_dados, 
    limpar_dados, 
    preparar_preprocessamento,
    separar_dados_treino_teste, 
    salvar_modelo_treinado, 
    FALHAS
)

def main():
    """Função principal - treina o modelo"""
    print("=" * 50)
    print("INICIANDO TREINAMENTO DO MODELO")
    print("=" * 50)
    
    # Marca tempo inicial
    inicio = time.time()
    
    # 1. Carregar dados
    print("\n1. 🗂️  Carregando dados...")
    dados = carregar_dados("data/Bootcamp_train.csv")
    
    if dados is None:
        print("❌ Não consegui carregar os dados de treino")
        print("💡 Coloque o arquivo Bootcamp_train.csv na pasta data/")
        return
    
    print(f"✅ Dados carregados: {len(dados)} linhas, {len(dados.columns)} colunas")
    
    # 2. Limpar dados
    print("\n2. 🧹 Limpando dados...")
    dados_limpos = limpar_dados(dados)
    
    if dados_limpos is None:
        print("❌ Problema na limpeza dos dados")
        return
    
    print(f"✅ Dados limpos: {len(dados_limpos)} linhas")
    
    # 3. Separar em treino e teste
    print("\n3. ✂️  Separando em treino e teste...")
    X_treino, X_teste, y_treino, y_teste = separar_dados_treino_teste(dados_limpos)
    
    if y_treino is None:
        print("❌ Dados não contém as colunas de falha")
        return
    
    print(f"✅ Separado: {len(X_treino)} treino, {len(X_teste)} teste")
    
    # 4. Preparar preprocessamento
    print("\n4. ⚙️  Preparando preprocessamento...")
    preprocessador, cols_num, cols_cat = preparar_preprocessamento(dados_limpos)
    
    if preprocessador is None:
        print("❌ Problema no preprocessamento")
        return
    
    print(f"✅ Preprocessador pronto: {len(cols_num)} numéricas, {len(cols_cat)} categóricas")
    
    # 5. Criar e treinar modelo
    print("\n5. 🤖 Criando e treinando modelo...")
    
    # RandomForest - parece ser bom para este tipo de problema
    # 100 árvores é um bom começo (não muito devagar)
    modelo_base = RandomForestClassifier(
        n_estimators=100,
        random_state=42,  # Para reproducibilidade
        n_jobs=-1,       # Usa todos os cores
        verbose=1        # Mostra progresso
    )
    
    # OneVsRest porque é multilabel (várias falhas ao mesmo tempo)
    modelo = OneVsRestClassifier(modelo_base)
    
    # Pipeline com preprocessamento + modelo
    pipeline = Pipeline([
        ('preprocessador', preprocessador),
        ('modelo', modelo)
    ])
    
    print("⏳ Treinando... (pode demorar alguns minutos)")
    pipeline.fit(X_treino, y_treino)
    print("✅ Modelo treinado!")
    
    # 6. Avaliar o modelo
    print("\n6. 📊 Avaliando modelo...")
    
    # Previsões no conjunto de teste
    y_pred = pipeline.predict(X_teste)
    
    # Acurácia geral
    acc = accuracy_score(y_teste, y_pred)
    print(f"📈 Acurácia geral: {acc:.3f}")
    
    # Relatório detalhado
    print("\n📋 Relatório por classe:")
    print(classification_report(y_teste, y_pred, target_names=FALHAS))
    
    # 7. Salvar modelo
    print("\n7. 💾 Salvando modelo...")
    sucesso = salvar_modelo_treinado(pipeline, "modelos/meu_modelo_treinado.joblib")
    
    if sucesso:
        print("✅ Modelo salvo com sucesso!")
    else:
        print("❌ Erro ao salvar modelo")
    
    # Tempo total
    fim = time.time()
    tempo_total = fim - inicio
    print(f"\n⏰ Tempo total: {tempo_total:.1f} segundos")
    
    print("\n" + "=" * 50)
    print("TREINAMENTO CONCLUÍDO! 🎉")
    print("=" * 50)
    
    print("""
    Próximos passos:
    1. Execute: streamlit run app.py
    2. Carregue o modelo na aplicação
    3. Faça previsões!
    """)

if __name__ == "__main__":
    # Mensagem simpática
    print("👋 Olá! Vamos treinar um modelo de machine learning!")
    print("💡 Certifique-se de ter os dados na pasta data/")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        print("😅 Acontece... tente novamente!")