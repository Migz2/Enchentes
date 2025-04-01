import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from loguru import logger

def train_model(data_path):
    """Carrega os dados processados, treina o modelo XGBoost e retorna o modelo treinado."""
    
    # 1. Carregar os dados com as features já criadas
    logger.info(f"Carregando dados de {data_path}")
    try:
        df = pd.read_csv(data_path)
        logger.success(f"Dados carregados com sucesso. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        raise

    # 2. Definir features (X) e target (y)
    logger.info("Preparando features e target")
    X = df.drop(columns=['Nível'])  # Exclui a coluna alvo
    y = df['Nível']                 # Define a variável alvo
    logger.info(f"Features disponíveis: {list(X.columns)}")

    # 3. Separar dados em treino e teste
    logger.info("Separando dados em treino e teste")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Dimensões dos dados - X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 4. Criar e treinar o modelo XGBoost
    logger.info("Iniciando treinamento do modelo XGBoost")
    try:
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.success("Modelo XGBoost treinado com sucesso")
    except Exception as e:
        logger.error(f"Erro durante o treinamento do modelo: {str(e)}")
        raise

    return model, X_test, y_test