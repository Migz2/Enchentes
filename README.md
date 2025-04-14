# Previsão de Enchentes - Rio Itajaí-Açu

Sistema de previsão do nível do Rio Itajaí-Açu utilizando Machine Learning para auxiliar na prevenção de enchentes em Rio do Sul, Santa Catarina.

## 🎯 Objetivo

O projeto visa desenvolver um modelo de Machine Learning capaz de prever o nível do Rio Itajaí-Açu com base em dados históricos meteorológicos e de nível do rio. Rio do Sul possui um histórico significativo de enchentes que impactam diretamente a vida da população, e este sistema busca fornecer previsões precisas para auxiliar na tomada de decisões e prevenção de desastres.

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **Jupyter Notebook**: Ambiente de desenvolvimento e análise de dados
- **Pandas & NumPy**: Manipulação e análise de dados
- **Scikit-learn**: Desenvolvimento de modelos de Machine Learning
- **XGBoost**: Algoritmo de gradient boosting para modelagem preditiva
- **Requests**: Coleta de dados via APIs
- **Loguru**: Sistema de logging
- **TQDM**: Barras de progresso para processos longos

## 📊 Fontes de Dados

- [Open Meteo API](https://open-meteo.com/): Dados meteorológicos históricos
- [Defesa Civil de Rio do Sul](https://defesacivil.riodosul.sc.gov.br/): Dados históricos do nível do rio

## 🚀 Como Executar

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
cd Enchentes
```

2. Crie um ambiente virtual e ative-o:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Execute o Jupyter Notebook:
```bash
jupyter notebook
```

5. Abra o notebook `notebooks/main.ipynb`

## 📝 Status do Projeto

### Concluído ✅
- Coleta de dados via API e Webscraping

### Em Andamento ⌛️
- Análise Exploratória das Enchentes nos últimos 6 anos
- Feature Engineering
- Treinamento do Modelo
- Avaliação de performance
- Deploy via API
- Website com visualização das previsões

## 🔜 Próximos Passos

1. **Análise Exploratória**
   - Desenvolver relatório com indicadores sobre enchentes
   - Criar visualizações e gráficos informativos
   - Identificar padrões e sazonalidades

2. **Feature Engineering**
   - Criar novas features relevantes
   - Incorporar dados adicionais do Open Meteo
   - Análise de correlações e importância de variáveis

3. **Modelagem**
   - Comparar performance entre diferentes algoritmos
   - Realizar tuning de hiperparâmetros
   - Implementar validação cruzada
   - Definir melhor horizonte de previsão

4. **Deploy**
   - Desenvolver API para servir as previsões
   - Criar interface web com visualizações
   - Implementar sistema de monitoramento em tempo real
