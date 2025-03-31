import joblib

def save_model(model, file_path="model.pkl"):
    """Salva o modelo treinado em um arquivo."""
    joblib.dump(model, file_path)
    print(f"Modelo salvo em {file_path}")

def load_model(file_path="model.pkl"):
    """Carrega um modelo previamente salvo."""
    return joblib.load(file_path)
