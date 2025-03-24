import pandas as pd
import os

def ordenar_csv_por_data(csv_destino, coluna_data):
    # Verifica se o arquivo existe
    if not os.path.exists(csv_destino):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_destino}")

    # Carrega os dados do CSV
    df = pd.read_csv(csv_destino)

    # Converte a coluna de data para datetime (para garantir ordenação correta)
    df[coluna_data] = pd.to_datetime(df[coluna_data], dayfirst=True, errors='coerce')

    # Ordena pela data em ordem crescente
    df = df.sort_values(by=coluna_data, ascending=True)

    # Salva as alterações no próprio arquivo
    df.to_csv(csv_destino, index=False, encoding='utf-8')

    print(f"Arquivo '{csv_destino}' ordenado com sucesso pela coluna '{coluna_data}'!")


def remover_coluna(coluna_para_remover):
    """Remove uma coluna específica de um arquivo CSV e salva a alteração."""
    
    # Carrega o arquivo CSV
    df = pd.read_csv("../dados_ench/web_scraping/web_scraping_main.csv")
    
    # Verifica se a coluna existe no CSV
    if coluna_para_remover in df.columns:
        df.drop(columns=[coluna_para_remover], inplace=True)  # Remove a coluna
        df.to_csv("../dados_ench/web_scraping/web_scraping_main.csv", index=False)  # Salva o CSV sem a coluna removida
        print(f"Coluna '{coluna_para_remover}' removida com sucesso.")
    else:
        print(f"A coluna '{coluna_para_remover}' não foi encontrada no arquivo.")


def adicionar_csv(origem_csv, destino_csv):
    
    try:
        # Carregar os dados do CSV de origem
        df_origem = pd.read_csv(origem_csv, sep = ",")
        if df_origem.empty:
            print(f"Erro: O arquivo '{origem_csv}' está vazio.")
            return
    except FileNotFoundError:
        print(f"Erro: O arquivo '{origem_csv}' não foi encontrado.")
        return

    try:
        # Carregar os dados do CSV de destino (se existir)
        df_destino = pd.read_csv(destino_csv, sep = ",")
    except FileNotFoundError:
        print("Destino não encontrado.!")

    # Adicionar os dados
    df_final = pd.concat([df_destino, df_origem], axis=1)

    # Salvar os dados atualizados no arquivo de destino
    df_final.to_csv(destino_csv, index=False, encoding='utf-8')

    print(f"Dados de '{origem_csv}' adicionados a '{destino_csv}' com sucesso!")


def ajustar_datas():
    df = pd.read_csv("../dados_ench/API/api_ench.csv")
    df["date"] = pd.to_datetime(df["date"], dayfirst=False)  # Converte corretamente, sem especificar formato
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M")  # Formata como 'YYYY-MM-DD HH:MM'
    # Salvar os DataFrames modificados de volta nos arquivos CSV
    df.to_csv("../dados_ench/API/api_ench.csv", index=False)


# Caminho do arquivo que será modificado (api_ench.csv)
base_dir = os.path.dirname(os.path.abspath(__file__))  # Diretório do script preparar_dados.py
csv_destino = os.path.join(base_dir, "../dados_ench/API/api_ench.csv")  # Caminho correto


# Executa a função
coluna_data = "Data"
csv_destino = "../dados_ench/web_scraping/web_scraping_main.csv"
ordenar_csv_por_data(csv_destino, coluna_data)
coluna_para_remover = "Data"
remover_coluna(coluna_para_remover)
origem_csv = "../dados_ench/web_scraping/web_scraping_main.csv"
destino_csv = "../dados_ench/API/api_ench.csv"
adicionar_csv(origem_csv, destino_csv)
ajustar_datas()