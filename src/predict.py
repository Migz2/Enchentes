import pandas as pd
import pickle
from pathlib import Path
from loguru import logger

# Configure logger
logger.add("../logs/predictions.log", rotation="500 MB", level="INFO")

MODEL_PATH = Path(__file__).parents[1] / "model" / "model.pkl"
DATA_PATH = Path(__file__).parents[1] / "data" / "api_recent" / "dados_api.csv"
OUTPUT_PATH = Path(__file__).parents[1] / "data" / "api_recent" / "previsoes.csv"

def carregar_modelo(model_path):
    logger.info(f"Carregando modelo de {model_path}")
    try:
        with open(model_path, 'rb') as file:
            modelo = pickle.load(file)
        logger.success("Modelo carregado com sucesso")
        return modelo
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        raise

def carregar_dados(data_path):
    logger.info(f"Carregando dados de {data_path}")
    try:
        dados = pd.read_csv(data_path, index_col="date")

        logger.success(f"Dados carregados com sucesso. Shape: {dados.shape}")
        return dados
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        raise

def fazer_previsao(modelo, dados):
    logger.info("Realizando previsões...")
    try:
        previsoes = modelo.predict(dados)
        logger.success(f"Previsões realizadas com sucesso. Total de previsões: {len(previsoes)}")
        return previsoes
    except Exception as e:
        logger.error(f"Erro ao fazer previsões: {str(e)}")
        raise

def salvar_previsoes(previsoes, output_path):
    logger.info(f"Salvando previsões em {output_path}")
    try:
        df = pd.DataFrame(previsoes, columns=['Previsao'])
        df.to_csv(output_path, index=False)
        logger.success("Previsões salvas com sucesso")
    except Exception as e:
        logger.error(f"Erro ao salvar previsões: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Iniciando processo de previsão")
    try:
        modelo = carregar_modelo(MODEL_PATH)
        novos_dados = carregar_dados(DATA_PATH)
        previsoes = fazer_previsao(modelo, novos_dados)
        salvar_previsoes(previsoes, OUTPUT_PATH)
        logger.success(f"Processo concluído. Previsões salvas em {OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Erro durante o processo de previsão: {str(e)}")
        raise