import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL do site com os dados
data_ini1 = "10%2F07%2F2024" #Padrão: "17%2F03%2F2025"
data_ini2 = "2024-07-10" #Padrão: "2025-03-17"
data_fin2 = "2024-07-13" #Padrão: "2025-03-17"
data_fin1 = "13%2F07%2F2024" #Padrão: "17%2F03%2F2025"
url = f"https://defesacivil.riodosul.sc.gov.br/index.php?r=externo%2Fmetragem-sensores&data_inicial-dreiksearch-data_inicial-disp={data_ini1}&DreikSearch%5Bdata_inicial%5D={data_ini2}&DreikSearch%5Bdata_final%5D={data_fin2}&DreikSearch%5Bintervalo%5D=60&DreikSearch%5Bordenacao%5D=3&data_final-dreiksearch-data_final-disp={data_fin1}&_tog1149016d=all&_pjax=%23kv-pjax-container-metragem-sensores"
# Simular um navegador
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

# Fazer a requisição
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
    
# Encontre a tabela correta no site
tabela = soup.find("table")

# Criar lista para armazenar os dados
dados = []

# Extrair cabeçalhos da tabela
headers = [th.text.strip() for th in tabela.find_all("th")]

# Extrair linhas da tabela
for row in tabela.find_all("tr")[1:]:  # Ignora o cabeçalho
    cols = [td.text.strip() for td in row.find_all("td")]
    if cols:
        dados.append(cols)

# Criar DataFrame com Pandas
df = pd.DataFrame(dados, columns=headers)

# Selecionar apenas a coluna 'Nível'
df_nivel = df[['Data','Nível']]

# Salvar os dados da coluna 'Nível' em um CSV
df_nivel.to_csv("nivel_extraido.csv", index=False, encoding="utf-8")

print("Coluna 'Nível' extraída com sucesso e salva em 'nivel_extraido.csv'")