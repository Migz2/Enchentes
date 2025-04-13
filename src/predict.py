import pandas as pd
import pickle
from pathlib import Path
from loguru import logger
import sys
sys.path.append(str(Path(__file__).parents[1]))
from model.feature_engineering import load_csv, calcular_lag, calcular_media_movel

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
        dados = load_csv(data_path)
        cols_numericas = ["temperature_2m","relative_humidity_2m","rain","cloud_cover","wind_speed_10m"]
        dados = dados[cols_numericas]
        logger.info(f"Dados carregados com sucesso. Shape: {dados.shape}")
        logger.info(f"Dados: {dados.head()}")
        janelas = [4, 8, 12, 24]
        for col in cols_numericas:
            for janela in janelas:
                dados[f"{col}_media_movel_{janela}h"] = calcular_media_movel(dados, col, janela)
                dados[f"{col}_lag_{janela}"] = calcular_lag(dados, col, janela)
        logger.success(f"Dados carregados com sucesso. Shape: {dados.shape}")
        logger.info(f"Dados: {dados.head()}")
        logger.info(f"Colunas disponíveis: {dados.columns.tolist()}")
        return dados
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        raise

def fazer_previsao(modelo, dados):
    logger.info("Realizando previsões...")
    try:
        # dataframe -> date, previsao
        previsoes_df = pd.DataFrame()
        previsoes_df['date'] = dados.index
        previsoes_df['previsao'] = modelo.predict(dados)
        logger.success(f"Previsões realizadas com sucesso. Total de previsões: {len(previsoes_df)}")
        return previsoes_df
    except Exception as e:
        logger.error(f"Erro ao fazer previsões: {str(e)}")
        raise

def salvar_previsoes(previsoes, output_path):
    logger.info(f"Salvando previsões em {output_path}")
    try:
        previsoes.to_csv(output_path, index=False)
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