# MLOps - Amphibians Classification

A comprehensive MLOps implementation for amphibian species classification using the UCI Machine Learning Repository dataset. This project demonstrates best practices in machine learning operations, from data preprocessing to model deployment.


<div align="center">
  <img src="img/photo-1615863.jpg" width="75%"/>
</div>

---

## Table of Contents

1. [Project Overview](#project-overview)
   - [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Running the Pipeline](#running-the-pipeline)
5. [Docker Deployment](#docker-deployment)
6. [MLflow Tracking](#mlflow-tracking)
7. [Data Versioning with DVC](#data-versioning-with-dvc)
8. [Testing and Governance](#testing-and-governance)
9. [Configuration](#configuration)
10. [Contributing](#contributing)
11. [License](#license)

---

## Project Overview

This project implements an end-to-end machine learning pipeline for classifying amphibian species. It incorporates MLOps best practices including experiment tracking, reproducible workflows, and containerized deployment.  The primary goals include:

- Automating repetitive tasks to boost productivity
- Ensuring reproducibility of model training and deployment
- Enabling scalability with tools such as Docker, MLflow, and logging frameworks

### Project Highlights
- **Data Ingestion**: Load and preprocess data seamlessly.
- **Data Preprocessing**: Use `DataPreprocessing` to handle data transformations and feature engineering.
- **Model Training and Tuning**: Implement custom training pipelines, including hyperparameter tuning and cross-validation.
- **Model Serving**: Support for deployment on various platforms.
- **Monitoring and Logging**: Track experiments and log metrics with MLflow.
- **Scalability**: Use Docker for containerization, making the pipeline reproducible and scalable.

### Dataset
- **Source**: [UCI Machine Learning Repository - Amphibians Dataset](https://archive.ics.uci.edu/dataset/528/amphibians)
- **Objective**: Classify amphibian species based on ecological and morphological features
- **Type**: Multiclass classification

---

## Project Structure

```
MLOps/
├── data/                      # Data directory
│   └── extracted/             # Extracted dataset files
│
├── mlops/                     # Core MLOps implementation
│   ├── config/                # Configuration files
│   ├── dataset/               # Dataset handling
│   │   ├── data_loader.py     # Data loading utilities
│   │   └── __init__.py
│   ├── feature_engineering/   # Feature processing
│   │   ├── data_preprocessing.py
│   │   └── __init__.py
│   ├── modeling/              # Model training and prediction
│   │   ├── prediction.py      # Prediction utilities
│   │   ├── train_model.py     # Training pipeline
│   │   └── __init__.py
│   ├── utils/                 # Helper utilities
│   │   ├── utils.py
│   │   └── __init__.py
│   ├── __init__.py
│   └── main.py                # Main execution script
│
├── models/                    # Saved model artifacts
├── notebooks/                 # Jupyter notebooks for exploration
├── reports/                   # Generated analysis reports
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Features

- 🔄 **Automated Pipeline**: End-to-end ML pipeline from data ingestion to model deployment
- 📊 **MLflow Integration**: Experiment tracking and model versioning
- 🐳 **Docker Support**: Containerized development and deployment
- 🔧 **Configurable**: Flexible configuration using YAML files
- 📈 **Model Monitoring**: Performance tracking and analysis

---

## Getting Started

### Prerequisites

- Python 3.8+
- Docker (optional)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/armandoBringas/MLOps.git
cd MLOps
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Pipeline

1. Prepare your data:
```bash
python mlops/dataset/data_loader.py
```

2. Train the model:
```bash
python mlops/main.py train
```

3. Make predictions:
```bash
python mlops/main.py predict --input_data path/to/data
```

---

## Docker Deployment

To build and deploy the project with Docker, follow these steps:

Build the Docker image:
```bash
docker build -t amphibians-classification .
```

Run the container:
```bash
docker run -p 5000:5000 amphibians-classification
```

Alternatively, We can use Docker Compose for a streamlined setup:

Build and run the container:
```bash
docker-compose up --build
```
Both methods will take some time to install all the necessary libraries. Once the installation is complete, use the following commands to access the container and start the main script with the specified Gin configuration:

```bash
docker exec -it mlops-mlops_service-1 /bin/bash
PYTHONPATH=$(pwd) python mlops/main.py -c "mlops/config/config.gin"
```
Note: You can adjust the ports as needed; some antivirus systems may use port 5000.

The application will be available on http://localhost:5002.
Note: When using Mac, is important to turn-off Hand-off and AirDrop, because those tools use the port 5000, so it the code will not be able to run properly
---

## MLflow Tracking

To track experiments using MLflow:

Access MLflow UI for experiment tracking:
```bash
mlflow ui --backend-store-uri file:///path/to/MLOps/mlruns
```

The MLflow UI will be available at http://localhost:5000.

All experiments, metrics, and models will be logged under the mlruns/ directory for easy tracking and comparison.
---

## Data Versioning with DVC

This project uses DVC to version control the datasets, making it easy to track and reproduce results.

Initialize DVC in the repository:
```bash
dvc init
```

Add dataset to DVC:
```bash
dvc add data/extracted/amphibians.csv
```

Commit changes and push to the remote repository:
```bash
git add data/extracted/amphibians.csv.dvc .dvc/config
git commit -m "Add dataset to DVC"
dvc push
```
To retrieve the dataset, use:
```bash
dvc pull
```
---

## Testing and Governance

This project follows governance and best practices for reproducibility and model reliability.
Governance Practices

    Code Standards: Follows PEP 8 and includes detailed documentation.
    Model Governance: Each model is versioned, logged, and evaluated for ethical considerations, reliability, and compliance with standards.

Unit and Integration Testing

    Unit Testing: Each component undergoes unit testing to ensure it functions independently.
    Integration Testing: Integration tests validate the interaction between modules in the ML pipeline.

Ensuring Reproducibility

    Data Versioning: DVC ensures reproducibility of dataset versions.
    Pipeline Automation: Automated ML pipelines ensure each stage is consistently reproducible across environments.
---

## Configuration

The project uses multiple configuration files to manage different aspects of the ML pipeline:

### Model Configuration (`model_config.yaml`)

Defines the models and their hyperparameters for training:

```yaml
models:
  # Logistic Regression
  Logistic Regression:
    class: sklearn.linear_model.LogisticRegression
    hyperparameters:
      C: [0.1, 1, 10]
      solver: ['liblinear', 'saga']
      class_weight: ['balanced']
      max_iter: [100, 200, 500, 1000]
  
  # Random Forest
  Random Forest:
    class: sklearn.ensemble.RandomForestClassifier
    hyperparameters:
      n_estimators: [50, 100, 200]
      max_depth: [5, 10, null]
      min_samples_split: [2, 5]
      min_samples_leaf: [1, 2]
      class_weight: ['balanced', 'balanced_subsample']
  
  # [Other models configuration...]

training:
  test_size: 0.3
  random_state: 42
  cv_folds: 5
```

### Pipeline Configuration (`config.gin`)

Controls the overall pipeline settings, including paths and MLflow configuration:

```python
# Directories
configure_logging.reports_dir = "reports"
save_model.models_dir = "models"

# MLflow settings
configure_mlflow.experiment_name = "Amphibians_Classification"
configure_mlflow.tracking_uri = "reports/mlruns"

# Data Loading
DataLoader.base_dir = '.'
DataLoader.data_url = 'https://archive.ics.uci.edu/static/public/528/amphibians.zip'
DataLoader.local_zip_path = 'data/amphibians.zip'
DataLoader.extract_dir = 'data/extracted'
DataLoader.extracted_file_name = 'amphibians.csv'
DataLoader.delimiter = ';'
DataLoader.encoding = 'utf-8'
DataLoader.error_bad_lines = True
```

### Configuration Files Structure

```
MLOps/
├── mlops/
│   └── config/
│       ├── model_config.yaml  # Model and hyperparameter configurations
│       └── config.gin         # Pipeline and environment configurations
```

### Key Configuration Features

1. **Model Configuration**:
   - Multiple model definitions with their hyperparameter search spaces
   - Training parameters (test split, cross-validation folds)
   - Easy to add or modify models and their hyperparameters

2. **Pipeline Configuration**:
   - Directory paths for logs, models, and reports
   - MLflow experiment tracking settings
   - Data loading parameters and paths
   - File handling configurations

### Using Configurations

To modify the pipeline behavior:

1. **Adjust Model Settings**:
   - Edit `model_config.yaml` to modify model hyperparameters
   - Add new models or remove existing ones
   - Modify training parameters

2. **Change Pipeline Settings**:
   - Update `config.gin` to modify paths and environment settings
   - Configure MLflow tracking
   - Adjust data loading parameters

---

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/NewFeature`
3. Commit your changes: `git commit -m 'Add NewFeature'`
4. Push to the branch: `git push origin feature/NewFeature`
5. Open a pull request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.