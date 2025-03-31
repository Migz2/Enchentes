from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    """Avalia o modelo treinado e imprime as métricas de erro."""
    
    y_pred = model.predict(X_test)

    # Calcular métricas de erro
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
