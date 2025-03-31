import pandas as pd
import pickle

MODEL_PATH = "../model/model.pkl"
DATA_PATH =  "../data/api_recent/dados_api.csv"
OUTPUT_PATH = "../data/api_recent/previsoes.csv"

def carregar_modelo(model_path):
    with open(model_path, 'rb') as file:
        modelo = pickle.load(file)
    return modelo

def carregar_dados(data_path):
    return pd.read_csv(data_path, index_col="date")

def fazer_previsao(modelo, dados):
    previsoes = modelo.predict(dados)
    return previsoes

def salvar_previsoes(previsoes, output_path):
    df = pd.DataFrame(previsoes, columns=['Previsao'])
    df.to_csv(output_path, index = False)

if __name__ == "__main__":
    modelo = carregar_modelo(MODEL_PATH)
    novos_dados = carregar_dados(DATA_PATH)
    previsoes = fazer_previsao(modelo, novos_dados)
    salvar_previsoes(previsoes, OUTPUT_PATH)

    print(f"Previsões salvas em {OUTPUT_PATH}")