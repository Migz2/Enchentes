from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from loguru import logger
import pandas as pd

def tune_hyperparameters(X_train, y_train):
    """Faz ajuste fino dos hiperparâmetros do XGBoost usando GridSearchCV."""
    
    logger.info("Iniciando processo de otimização de hiperparâmetros")
    
    try:
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        logger.info("Modelo base XGBoost criado")

        # Definir o grid de parâmetros
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
        logger.info(f"Grid de hiperparâmetros definido: {param_grid}")

        # Usar GridSearchCV para encontrar os melhores hiperparâmetros
        logger.info("Iniciando busca em grid com validação cruzada")
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3)
        grid_search.fit(X_train, y_train)

        # Logging dos resultados
        logger.info("Resultados da otimização:")
        logger.info(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
        logger.info(f"Melhor score: {-grid_search.best_score_:.4f} (MSE)")
        
        # Log top 3 combinações
        results_df = pd.DataFrame(grid_search.cv_results_)
        top_3 = results_df.nlargest(3, 'mean_test_score')
        for i, row in top_3.iterrows():
            logger.info(f"Top {i+1} combinação:")
            logger.info(f"Parâmetros: {row['params']}")
            logger.info(f"Score médio: {-row['mean_test_score']:.4f} (MSE)")

        logger.success("Otimização de hiperparâmetros concluída com sucesso")
        return grid_search.best_estimator_
    
    except Exception as e:
        logger.error(f"Erro durante a otimização de hiperparâmetros: {str(e)}")
        raise
