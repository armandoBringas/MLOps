import pytest
from mlops.dataset.data_loader import DataLoader

def test_load_data():
    # Instancia la clase DataLoader
    data_loader = DataLoader(
        data_url="https://archive.ics.uci.edu/static/public/528/amphibians.zip",
        local_csv_path="extracted/amphibians.csv",
        extracted_file_name="amphibians.csv"  # Agrega el nombre del archivo extraído
    )

    # Llama al método load para cargar los datos
    data = data_loader.load()
    assert data is not None, "El conjunto de datos debería estar cargado"
    assert len(data) > 0, "El conjunto de datos debería tener registros"
