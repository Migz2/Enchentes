from pathlib import Path
import joblib
from loguru import logger
import os

MODEL_PATH = Path(__file__).parents[1] / "model" / "model.pkl"

def save_model(model, file_path=MODEL_PATH):
    """Salva o modelo treinado em um arquivo."""
    logger.info(f"Iniciando salvamento do modelo em {file_path}")
    try:
        # Garantir que o diretório existe
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Salvar o modelo
        joblib.dump(model, file_path)
        
        # Verificar tamanho do arquivo
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Tamanho em MB
        logger.success(f"Modelo salvo com sucesso em {file_path} (Tamanho: {file_size:.2f} MB)")
    except Exception as e:
        logger.error(f"Erro ao salvar o modelo: {str(e)}")
        raise

def load_model(file_path=MODEL_PATH):
    """Carrega um modelo previamente salvo."""
    logger.info(f"Tentando carregar modelo de {file_path}")
    try:
        if not os.path.exists(file_path):
            logger.error(f"Arquivo do modelo não encontrado: {file_path}")
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
            
        model = joblib.load(file_path)
        logger.success(f"Modelo carregado com sucesso de {file_path}")
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo: {str(e)}")
        raise
