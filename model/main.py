from train_model import train_model
from hyperparameter_tuning import tune_hyperparameters
from evaluate import evaluate_model
from save_model import save_model

DATA_PATH = "../data/dados_ench/enchentes/enchentes.csv"

# 1. Treinar modelo inicial
model, X_test, y_test = train_model(DATA_PATH)

# 2. Ajustar hiperparâmetros
best_model = tune_hyperparameters(X_test, y_test)

# 3. Avaliar modelo final
evaluate_model(best_model, X_test, y_test)

# 4. Salvar modelo final
save_model(best_model, "model.pkl")
