from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from loguru import logger

def evaluate_model(model, X_test, y_test):
    """Avalia o modelo treinado e imprime as métricas de erro."""
    
    logger.info("Iniciando avaliação do modelo")
    try:
        y_pred = model.predict(X_test)
        logger.info("Previsões realizadas no conjunto de teste")

        # Calcular métricas de erro
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        logger.info("Métricas de avaliação:")
        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"R²: {r2:.2f}")

        # Análise adicional
        erro_medio = np.mean(np.abs(y_test - y_pred))
        erro_max = np.max(np.abs(y_test - y_pred))
        logger.info(f"Erro médio absoluto: {erro_medio:.2f}")
        logger.info(f"Erro máximo absoluto: {erro_max:.2f}")

        logger.success("Avaliação do modelo concluída com sucesso")
    except Exception as e:
        logger.error(f"Erro durante a avaliação do modelo: {str(e)}")
        raise
