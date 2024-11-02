import pytest
from mlops.main import run

def test_pipeline():
    # Configura la ruta de configuración para el pipeline
    config_path = "/app/mlops/config/config.gin"

    # Ejecuta el pipeline completo
    try:
        result = run(config_path=config_path)
        assert result is not None, "El pipeline debería retornar resultados"
    except Exception as e:
        pytest.fail(f"El pipeline falló con una excepción: {e}")
