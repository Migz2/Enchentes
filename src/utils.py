import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("../data/api_recent/dados_api.csv")
    return df