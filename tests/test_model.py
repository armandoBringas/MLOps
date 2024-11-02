import pytest
from mlops.dataset.data_loader import DataLoader
from mlops.modeling.train_model import TrainModel

def test_model_training():
    data_loader = DataLoader(
        data_url="https://archive.ics.uci.edu/static/public/528/amphibians.zip",
        local_csv_path="extracted/amphibians.csv",
        extracted_file_name="amphibians.csv"
    )
    data = data_loader.load()
    X = data[['Motorway', 'SR', 'NR', 'TR', 'VR']]
    y = data['Common toad']
    
    model = TrainModel(X, y)
    result = model.train_model('random_forest')
    assert result is not None, "El entrenamiento del modelo debería retornar un resultado válido"

def test_model_accuracy():
    data_loader = DataLoader(
        data_url="https://archive.ics.uci.edu/static/public/528/amphibians.zip",
        local_csv_path="extracted/amphibians.csv",
        extracted_file_name="amphibians.csv"
    )
    data = data_loader.load()
    X = data[['Motorway', 'SR', 'NR', 'TR', 'VR']]
    y = data['Common toad']
    
    model = TrainModel(X, y)
    trained_model = model.train_model('random_forest')
    result = model.evaluate_model_performance(trained_model)
    assert result is not None, "La evaluación del modelo debería retornar un resultado válido"
