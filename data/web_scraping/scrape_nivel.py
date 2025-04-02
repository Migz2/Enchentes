import requests
from bs4 import BeautifulSoup
import pandas as pd
from loguru import logger
from pathlib import Path

class WebScraper:
    """Classe responsável por realizar o Web Scraping."""
    def __init__(self):
        self.url = "https://defesacivil.riodosul.sc.gov.br/index.php?r=externo%2Fmetragem-sensores&data_inicial-dreiksearch-data_inicial-disp={dia_ini1}&DreikSearch%5Bdata_inicial%5D={dia_ini2}&DreikSearch%5Bdata_final%5D={dia_fin2}&DreikSearch%5Bintervalo%5D=60&DreikSearch%5Bordenacao%5D=3&data_final-dreiksearch-data_final-disp={dia_fin1}&_tog1149016d=all&_pjax=%23kv-pjax-container-metragem-sensores&_pjax=%23kv-pjax-container-metragem-sensores"
        logger.info("WebScraper inicializado")

    def fetch_html(self, dia_ini1, dia_ini2, dia_fin1, dia_fin2):
        """Faz a requisição e retorna o HTML da página."""
        logger.debug(f"Buscando dados para o período: {dia_ini2} até {dia_fin2}")
        response = requests.get(self.url.format(dia_ini1=dia_ini1, dia_ini2=dia_ini2, dia_fin1=dia_fin1, dia_fin2=dia_fin2))
        if response.status_code == 200:
            logger.debug("Conteúdo HTML obtido com sucesso")
            return response.text
        else:
            logger.error(f"Falha ao buscar dados. Código de status: {response.status_code}")
            raise Exception(f"Erro ao acessar {self.url}: Código {response.status_code}")

    def parse_data(self, start_date, end_date):
        logger.info(f"Iniciando análise de dados de {start_date} até {end_date}")
        # Gera os intervalos de datas. O timeframe é de 3 dias
        tables = []
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        timeframe = 3
        total_intervals = len(date_range) // timeframe
        logger.info(f"Serão processados {total_intervals} intervalos de {timeframe} dias cada")

        # Calculate the number of complete intervals
        num_intervals = (len(date_range) + timeframe - 1) // timeframe
        
        for i in range(num_intervals):
            start_idx = len(date_range) - (i + 1) * timeframe
            end_idx = len(date_range) - i * timeframe
            date_range_slice = date_range[start_idx:end_idx]
            
            if len(date_range_slice) == 0:
                logger.warning("Intervalo de datas vazio encontrado, pulando")
                continue
                
            dia_ini2, dia_fin2 = date_range_slice[0], date_range_slice[-1]
            dia_ini1, dia_fin1 = dia_ini2.strftime('%d/%m/%Y').replace('/', '%2F'), dia_fin2.strftime('%d/%m/%Y').replace('/', '%2F')
            try:
                html = self.fetch_html(dia_ini1, dia_ini2, dia_fin1, dia_fin2)
                soup = BeautifulSoup(html, 'html.parser')
                table = soup.find('table')
                data = []
                for row in table.find_all('tr')[1:]: # Pula o cabeçalho
                    cols = row.find_all('td')
                    if cols:
                        date = cols[0].text.strip()
                        nivel = cols[1].text.strip()
                        data.append({"Data": date, "Nível": nivel})
                if not table:
                    logger.error("Tabela não encontrada no conteúdo HTML")
                    raise Exception("Tabela não encontrada no HTML!")
                tables.append(pd.DataFrame(data))
                logger.debug(f"Dados processados com sucesso para o período: {dia_ini2} até {dia_fin2}")
            except Exception as e:
                logger.error(f"Erro ao processar período {dia_ini2} até {dia_fin2}: {str(e)}")
                raise

        logger.info("Análise de dados concluída com sucesso")
        if not tables:
            logger.warning("Nenhum dado foi coletado")
            return pd.DataFrame(columns=["Data", "Nível"])
            
        df = pd.concat(tables, ignore_index=True)
        # Convert Data column to datetime
        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y %H:%M')
        # Convert Nível column to float
        df['Nível'] = pd.to_numeric(df['Nível'], errors='coerce')
        # Sort by date
        df = df.sort_values('Data', ascending=True).reset_index(drop=True)
        logger.info(f"Formato final do DataFrame: {df.shape}")
        return df

if __name__ == "__main__":
    # Configure loguru
    scraper = WebScraper()
    start_date = "2023-09-29"
    end_date = "2023-12-15"
    df = scraper.parse_data(start_date, end_date)
    logger.info(f"Formato final do DataFrame: {df.shape}")
    logger.info(df.head())

    # Salvar o DataFrame em um arquivo CSV
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"nivel_data_{start_date}_{end_date}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"DataFrame salvo em: {output_file}")
