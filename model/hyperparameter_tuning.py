from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def tune_hyperparameters(X_train, y_train):
    """Faz ajuste fino dos hiperparâmetros do XGBoost usando GridSearchCV."""
    
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Definir o grid de parâmetros
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }

    # Usar GridSearchCV para encontrar os melhores hiperparâmetros
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_
