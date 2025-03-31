import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_model(data_path):
    """Carrega os dados processados, treina o modelo XGBoost e retorna o modelo treinado."""
    
    # 1. Carregar os dados com as features já criadas
    df = pd.read_csv(data_path)


    # 2. Definir features (X) e target (y)
    X = df.drop(columns=['Nível', 'date'])  # Exclui a coluna alvo
    y = df['Nível']                 # Define a variável alvo
    print(X.head())


    # 3. Separar dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Criar e treinar o modelo XGBoost
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test