import os
import pandas as pd
import numpy as np
import plotly.express as px

def plotar_importancia_variaveis(modelo, variaveis):
    """Tenta extrair importâncias de atributos do modelo e retornar gráfico.`"""
    try:
        if hasattr(modelo, 'feature_importances_'):
            importancias = modelo.feature_importances_
            df_imp = pd.DataFrame({'variavel': variaveis, 'importancia': importancias})
            df_imp = df_imp.sort_values('importancia', ascending=True)
            fig = px.bar(df_imp, x='importancia', y='variavel', orientation='h', title='Importância das Variáveis')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            return fig
        # Se for modelo linear, tenta usar coeficientes
        if hasattr(modelo, 'coef_'):
            coef = np.ravel(modelo.coef_)
            df_imp = pd.DataFrame({'variavel': variaveis, 'importancia': np.abs(coef)})
            df_imp = df_imp.sort_values('importancia', ascending=True)
            return px.bar(df_imp, x='importancia', y='variavel', orientation='h', title='Importância (coef)')
    except Exception:
        pass
    return px.bar(title='Importância não disponível')

# ------------------------------
# Montagem do payload para a API de avaliação
# ------------------------------
def montar_payload_api(df_teste, df_predicoes, tipo_modelagem):
    payload = {'predictions': []}

    if df_predicoes is None:
        return payload

    # Garante coluna id
    if df_teste is not None and 'id' in df_teste.columns and 'id' in df_predicoes.columns:
        base = df_teste[['id']].merge(df_predicoes, on='id', how='left')
    else:
        base = df_predicoes.copy()
    if 'id' not in base.columns:
        base.insert(0, 'id', np.arange(len(base)))

    if tipo_modelagem.startswith('Binária'):
        for _, r in base.iterrows():
            payload['predictions'].append({
                'id': int(r['id']),
                'label': int(r.get('pred_qualquer_falha', 0)),
                'prob': float(r.get('proba_falha', 0.0))
            })
    else:
        for _, r in base.iterrows():
            payload['predictions'].append({
                'id': int(r['id']),
                'label': int(r.get('pred_tipo_falha_cod', 0))
            })

    return payload
