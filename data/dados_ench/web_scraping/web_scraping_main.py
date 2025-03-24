import requests
from bs4 import BeautifulSoup
import pandas as pd

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
        """Extrai os dados da coluna 'nível' do HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        tabela = soup.find('table')  # Supondo que os dados estejam em uma tabela
        if not tabela:
            raise Exception("Tabela não encontrada no HTML!")
        
        dados_nivel = []
        for linha in tabela.find_all('tr')[1:]:  # Pulando cabeçalho
            colunas = linha.find_all('td')
            if colunas:
                data = colunas[0].text.strip()
                nivel = colunas[1].text.strip()  # Pegando a coluna de Nível (Ajuste conforme necessário)
                dados_nivel.append({"Data": data, "Nível": nivel})        
        return dados_nivel

class CSVManager:
    """Classe para salvar os dados em um CSV."""
    @staticmethod
    def append_to_csv(data, filename="web_scraping_main.csv"):
        df = pd.DataFrame(data, columns=["Data", "Nível"])
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Dados salvos em {filename}")

if __name__ == "__main__":
    dia_ini1 = "29%2F09%2F2023" # Padrão: 04%2F07%2F2024
    dia_ini2 = "2023-09-29" # Padrão: 2024-07-04
    dia_fin1 = "02%2F10%2F2023" # Padrão: 04%2F07%2F2024
    dia_fin2 = "2023-10-02" # Padrão: 2024-07-04
    url = f"https://defesacivil.riodosul.sc.gov.br/index.php?r=externo%2Fmetragem-sensores&data_inicial-dreiksearch-data_inicial-disp={dia_ini1}&DreikSearch%5Bdata_inicial%5D={dia_ini2}&DreikSearch%5Bdata_final%5D={dia_fin2}&DreikSearch%5Bintervalo%5D=60&DreikSearch%5Bordenacao%5D=3&data_final-dreiksearch-data_final-disp={dia_fin1}&_tog1149016d=all&_pjax=%23kv-pjax-container-metragem-sensores&_pjax=%23kv-pjax-container-metragem-sensores"
    scraper = WebScraper(url)
    html_content = scraper.fetch_html()
    nivel_data = scraper.parse_data(html_content)    
    CSVManager.append_to_csv(nivel_data)
