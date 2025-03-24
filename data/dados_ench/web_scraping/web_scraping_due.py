import requests
from bs4 import BeautifulSoup
import pandas as pd
import os


class WebScraper:
    """Classe responsável por realizar o Web Scraping."""
    
    def __init__(self, url):
        self.url = url

    def fetch_html(self):
        """Faz a requisição e retorna o HTML da página."""
        response = requests.get(self.url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Erro ao acessar {self.url}: Código {response.status_code}")

    def parse_data(self, html):
        """Extrai os dados das colunas 'Data' e 'Nível' do HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        tabela = soup.find('table')  # Supondo que os dados estejam em uma tabela
        if not tabela:
            raise Exception("Tabela não encontrada no HTML!")

        dados = []
        for linha in tabela.find_all('tr')[1:]:  # Pulando cabeçalho
            colunas = linha.find_all('td')
            if colunas:
                data = colunas[0].text.strip()  # Pegando a coluna Data
                nivel = colunas[1].text.strip()  # Pegando a coluna Nível
                dados.append({"Data": data, "Nível": nivel})        
        return dados


class CSVManager:
    """Classe para gerenciar o arquivo CSV."""
    
    @staticmethod
    def append_to_csv(data, filename="web_scraping_main.csv"):
        """Adiciona novos dados ao CSV sem sobrescrever os antigos."""
        df_novos_dados = pd.DataFrame(data, columns=["Data", "Nível"])

        # Se o arquivo já existe, carrega os dados antigos
        if os.path.exists(filename):
            df_antigos = pd.read_csv(filename)
            df_final = pd.concat([df_antigos, df_novos_dados], ignore_index=True)
        else:
            df_final = df_novos_dados

        # Salva o arquivo atualizado
        df_final.to_csv(filename, index=False, encoding="utf-8")
        print(f"Dados adicionados ao {filename}")


if __name__ == "__main__":
    # Definição de datas no formato necessário
    dia_ini1 = "02%2F10%2F2023" #"11%2F07%2F2024"
    dia_ini2 = "2023-10-02"  #"2024-07-11"
    dia_fin1 = "02%2F10%2F2023" #"11%2F07%2F2024"
    dia_fin2 = "2023-10-02" #"2024-07-11"

    # URL formatada com as datas
    url = (f"https://defesacivil.riodosul.sc.gov.br/index.php?r=externo%2Fmetragem-sensores"
           f"&data_inicial-dreiksearch-data_inicial-disp={dia_ini1}"
           f"&DreikSearch%5Bdata_inicial%5D={dia_ini2}"
           f"&DreikSearch%5Bdata_final%5D={dia_fin2}"
           f"&DreikSearch%5Bintervalo%5D=60&DreikSearch%5Bordenacao%5D=3"
           f"&data_final-dreiksearch-data_final-disp={dia_fin1}"
           f"&_tog1149016d=all&_pjax=%23kv-pjax-container-metragem-sensores"
           f"&_pjax=%23kv-pjax-container-metragem-sensores")

    # Faz o scraping e adiciona os dados ao CSV
    scraper = WebScraper(url)
    html_content = scraper.fetch_html()
    nivel_data = scraper.parse_data(html_content)    
    CSVManager.append_to_csv(nivel_data)
