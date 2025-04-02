from pathlib import Path
import pandas as pd
from loguru import logger

DADOS_ENCH = Path(__file__).parents[1] / "data" / "dados_ench"

def load_csv(file_path):
    """Carrega o CSV e retorna um DataFrame ordenado pela data."""
    df = pd.read_csv(file_path, parse_dates=["date"])
    df = df.sort_values(by="date")  # Garante que os dados estejam ordenados
    df = df.set_index("date")
    return df

def copy_column(df, original_column, new_column):
    """Copia os valores de uma coluna para uma nova coluna."""
    df[new_column] = df[original_column]
    return df

def calcular_media_movel(df, coluna, janela):
    """Calcula a média móvel de uma coluna."""
    return df[coluna].rolling(window=janela).mean()

def calcular_lag(df, coluna, lag):
    """Calcula o lag de uma coluna."""
    return df[coluna].shift(lag)

def calcular_target(df, coluna, lag):
    """Calcula o target de uma coluna."""
    return df[coluna].shift(-lag)

if __name__ == "__main__":
    # Fluxo principal
    file_path = DADOS_ENCH / "API" / "api_ench.csv"  # Substitua pelo nome real do arquivo
    df = load_csv(file_path)
    logger.info(f"Dados carregados com sucesso. Shape: {df.shape}")
    logger.info(df.head())  # Mostra as primeiras linhas

    col_numericas = ["temperature_2m","relative_humidity_2m","rain","cloud_cover","wind_speed_10m"]
    horas_janela = [4, 8, 12, 24]
    for coluna in col_numericas:
        for janela in horas_janela:
            df[f"{coluna}_media_movel_{janela}h"] = calcular_media_movel(df, coluna, janela)
            df[f"{coluna}_lag_{janela}"] = calcular_lag(df, coluna, janela)

    df["target"] = calcular_target(df, "Nível", 3)

    df.to_csv(DADOS_ENCH / "API" / "api_ench_com_features.csv")
    logger.success("Dados com features salvo com sucesso")