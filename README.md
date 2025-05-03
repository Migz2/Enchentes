# Previsão de Enchentes - Rio Itajaí-Açu

Sistema de previsão do nível do Rio Itajaí-Açu utilizando Machine Learning para auxiliar na prevenção de enchentes em Rio do Sul, Santa Catarina.

## 🎯 Objetivo

O projeto visa desenvolver um modelo de Machine Learning capaz de prever o nível do Rio Itajaí-Açu com base em dados históricos meteorológicos e de nível do rio. Rio do Sul possui um histórico significativo de enchentes que impactam diretamente a vida da população, e este sistema busca fornecer previsões precisas para auxiliar na tomada de decisões e prevenção de desastres.

### Objetivos Específicos
- Prever o nível do rio com antecedência de até 72 horas
- Identificar padrões climáticos que precedem enchentes
- Fornecer alertas antecipados para a população
- Auxiliar órgãos públicos na tomada de decisão

## 🛠️ Tecnologias Utilizadas

- **Python 3.9.18**: Versão específica para compatibilidade com PyCaret
- **Gerenciamento de Dependências**:
  - `pyproject.toml`: Configuração moderna do projeto
  - `uv`: Gerenciador de pacotes Python rápido e eficiente

- **Análise de Dados e ML**:
  - `pandas` (1.3.0-2.0.0): Manipulação e análise de dados
  - `numpy` (1.21.0-1.24.0): Computação numérica
  - `scikit-learn` (1.0.0-1.3.0): Algoritmos de ML
  - `pycaret` (3.1.0): AutoML e experimentação rápida

- **Modelos Avançados**:
  - `xgboost`: Gradient boosting otimizado
  - `lightgbm`: Framework de gradient boosting leve
  - `catboost`: ML com suporte nativo a variáveis categóricas

- **Visualização e Interface**:
  - `plotly` (5.5.0-6.0.0): Gráficos interativos
  - `matplotlib` (3.3.0-3.8.0): Visualizações estáticas
  - `seaborn`: Visualizações estatísticas
  - `jupyter & jupyterlab`: Ambiente interativo

- **Utilitários**:
  - `loguru`: Logging avançado
  - `tqdm`: Barras de progresso
  - `joblib`: Paralelização e caching

## 📊 Fontes de Dados

### Open Meteo API
- Dados meteorológicos históricos horários
- Variáveis coletadas:
  - Temperatura
  - Precipitação
  - Umidade
  - Velocidade do vento
  - Pressão atmosférica

### Defesa Civil de Rio do Sul
- Medições do nível do rio a cada 15 minutos
- Histórico de enchentes desde 2017
- Marcas históricas e cotas de inundação

## 🚀 Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/Migz2/Enchentes.git
cd Enchentes
```

2. Instale o uv (gerenciador de pacotes Python rápido):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Crie um ambiente virtual e instale as dependências:
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# ou
.\.venv\Scripts\activate  # Windows
uv pip install -e .
```

4. Execute o Jupyter Lab:
```bash
jupyter lab
```

5. Abra o notebook `notebooks/main.ipynb`

## 📦 Estrutura do Projeto

```
Enchentes/
├── data/                   # Scripts de processamento de dados
│   ├── __init__.py        # Torna o diretório um módulo Python
│   ├── collect.py         # Scripts de coleta de dados
│   ├── process.py         # Processamento e limpeza
│   └── output/            # Dados processados
│
├── notebooks/             # Jupyter notebooks
│   ├── main.ipynb        # Notebook principal
│   └── eda.ipynb         # Análise exploratória
│
├── .gitignore            # Arquivos ignorados pelo git
├── pyproject.toml        # Configuração do projeto
├── README.md            # Documentação
└── logs.log            # Registro de execução
```

## 📝 Status do Projeto

### Concluído ✅
- Setup inicial do projeto com ambiente Python 3.9
- Configuração de dependências via `pyproject.toml`
- Integração com PyCaret 3.1.0
- Estrutura base do projeto

### Em Andamento ⌛️
- Coleta e processamento de dados históricos
- Análise exploratória das enchentes
- Desenvolvimento do modelo preditivo
- Implementação do pipeline de ML

### Planejado 📋
- API REST para servir previsões
- Interface web para visualização
- Sistema de alertas automáticos
- Documentação técnica detalhada

## 🔜 Próximos Passos

### 1. Análise Exploratória
- Análise temporal das enchentes
- Correlação entre variáveis meteorológicas
- Identificação de padrões sazonais
- Visualizações interativas com Plotly

### 2. Feature Engineering
- Criação de features temporais
- Agregações por diferentes janelas
- Incorporação de dados externos
- Seleção de features relevantes

### 3. Modelagem
- Experimentação com diferentes algoritmos
- Otimização de hiperparâmetros
- Validação temporal apropriada
- Análise de importância de features

### 4. Implementação
- Desenvolvimento da API
- Interface web responsiva
- Sistema de monitoramento
- Documentação da API

## 👥 Contribuição

Contribuições são bem-vindas! Por favor, siga estes passos:

1. Faça fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📧 Contato

Miguel de Almeida Silva - migs.asilva9@gmail.com

Link do projeto: [https://github.com/Migz2/Enchentes](https://github.com/Migz2/Enchentes)
