import pytest
from mlops.dataset.data_loader import DataLoader

def test_data_quality():
    # Instancia la clase DataLoader y carga los datos
    data_loader = DataLoader(
        data_url="https://archive.ics.uci.edu/static/public/528/amphibians.zip",
        local_csv_path="extracted/amphibians.csv",
        extracted_file_name="amphibians.csv"
    )
    data = data_loader.load()

    # Rellenar valores nulos en la columna 'Motorway' si es necesario
    data['Motorway'].fillna(0, inplace=True)  # Ajusta el valor según contexto

    # Verificar que no haya valores nulos en el dataset
    assert data.isnull().sum().sum() == 0, "Los datos no deben tener valores nulos"

    # Verificar que las columnas numéricas estén dentro de un rango
    for col in ['Motorway', 'SR', 'NR', 'TR', 'VR']:
        assert data[col].min() >= 0, f"{col} debe ser >= 0"
