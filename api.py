import requests
import pandas as pd
import json

def submit_to_api(csv_file_path, api_url):
    """
    Envia as previsões do arquivo Bootcamp_test.csv para a API de avaliação
    """
    # Carrega as previsões
    df_test = pd.read_csv(csv_file_path)
    
    # Prepara os dados no formato esperado pela API
    payload = {
        "predictions": df_test.to_dict(orient='records')
    }
    
    # Envia para a API
    response = requests.post(api_url, json=payload)
    
    if response.status_code == 200:
        print("✅ Previsões enviadas com sucesso!")
        print("Resultado:", response.json())
    else:
        print("❌ Erro ao enviar previsões:", response.text)

# Exemplo de uso
if __name__ == "__main__":
    API_URL = "https://api-bootcamp-cdia.herokuapp.com/evaluate"  # URL fornecida
    submit_to_api("data/Bootcamp_test_predictions.csv", API_URL)
