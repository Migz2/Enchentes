from pathlib import Path
import pandas as pd

DADOS_ENCH = Path(__file__).parents[1] / "data" / "dados_ench"

def load_csv(file_path):
    """Carrega o CSV e retorna um DataFrame ordenado pela data."""
    df = pd.read_csv(file_path, parse_dates=["date"])
    df = df.sort_values(by="date")  # Garante que os dados estejam ordenados
    return df

def copy_column(df, original_column, new_column):
    """Copia os valores de uma coluna para uma nova coluna."""
    df[new_column] = df[original_column]
    return df
# Fluxo principal
file_path = DADOS_ENCH / "enchentes" / "enchentes.csv"  # Substitua pelo nome real do arquivo

df = load_csv(file_path) 

df.to_csv(DADOS_ENCH / "enchentes" / "enchentes.csv")
print(df.head())  # Mostra as primeiras linhas
