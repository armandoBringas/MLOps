import pytest
import pandas as pd
from mlops.feature_engineering.data_preprocessing import DataPreprocessing

@pytest.fixture
def sample_data():
    # Datos de prueba simulados
    data = {
        'ID': [1, 2],  # Agregar ID para cumplir con el método clean_data
        'MV': ['A1', 'A2'],
        'SR': [600, 700],
        'NR': [1, 2],
        'TR': [1, 1],
        'VR': [4, 1],
        'SUR1': [6, 10],
        'SUR2': [2, 6],
        'SUR3': [10, 3],
        'UR': [0, 1],
        'FR': [0, 1],
        'OR': [50, 75],
        'RR': [0, 1],
        'BR': [0, 1],
        'MR': [0, 1],
        'CR': [1, 1],
        'Green frogs': [0, 1],
        'Brown frogs': [0, 1],
        'Common toad': [1, 0],
        'Fire-bellied toad': [0, 1],
        'Tree frog': [0, 0],
        'Common newt': [1, 1],
        'Great crested newt': [0, 0]
    }
    return pd.DataFrame(data)

def test_clean_data(sample_data):
    # Instancia de la clase DataPreprocessing y limpieza de datos
    preprocessor = DataPreprocessing(sample_data)
    clean_df = preprocessor.clean_data()
    
    # Verificar la limpieza y transformación de datos
    expected_columns = [
        'MV', 'SR', 'NR', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR', 'OR',
        'RR', 'BR', 'MR', 'CR', 'Green frogs', 'Brown frogs', 'Common toad',
        'Fire-bellied toad', 'Tree frog', 'Common newt', 'Great crested newt'
    ]
    assert list(clean_df.columns) == expected_columns, "Las columnas no coinciden con las esperadas después de la limpieza"

def test_normalize_data(sample_data):
    # Instancia de la clase DataPreprocessing y normalización de datos
    preprocessor = DataPreprocessing(sample_data)
    normalized_df = preprocessor.normalize_data()
    
    # Verifica que el número de columnas sea correcto tras la normalización
    assert len(normalized_df.columns) == len(sample_data.select_dtypes(include=['int64', 'float64']).columns), \
        "El número de columnas en el DataFrame normalizado no coincide con las columnas numéricas del original"

def test_apply_pca(sample_data):
    # Instancia de la clase DataPreprocessing, normalización y PCA
    preprocessor = DataPreprocessing(sample_data)
    X_normalized = preprocessor.normalize_data()
    X_pca = preprocessor.apply_pca(X_normalized)

    # Verifica que el PCA reduzca la dimensionalidad adecuadamente
    assert X_pca.shape[1] <= X_normalized.shape[1], "PCA debería reducir la dimensionalidad de los datos"