from pathlib import Path
from loguru import logger
from train_model import train_model
from hyperparameter_tuning import tune_hyperparameters
from evaluate import evaluate_model
from save_model import save_model

# Configure logger
logger.add("../logs/model_training.log", rotation="500 MB", level="INFO")

DATA_PATH = Path(__file__).parents[1] / "data" / "dados_ench" / "enchentes" / "enchentes.csv"

logger.info("Iniciando processo de treinamento do modelo")

# 1. Treinar modelo inicial
logger.info("Treinando modelo inicial...")
model, X_test, y_test = train_model(DATA_PATH)
logger.success("Modelo inicial treinado com sucesso")

# 2. Ajustar hiperparâmetros
logger.info("Iniciando ajuste de hiperparâmetros...")
best_model = tune_hyperparameters(X_test, y_test)
logger.success("Ajuste de hiperparâmetros concluído")

# 3. Avaliar modelo final
logger.info("Avaliando modelo final...")
evaluate_model(best_model, X_test, y_test)
logger.success("Avaliação do modelo concluída")

# 4. Salvar modelo final
logger.info("Salvando modelo final...")
save_model(best_model, "model.pkl")
logger.success("Modelo final salvo com sucesso")

logger.info("Processo de treinamento concluído")
