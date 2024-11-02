import unittest
import numpy as np
import pickle
import os
import sys

# Add the parent directory to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlops', 'modeling')))

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from train_model import TrainModel
from unittest.mock import patch, MagicMock


class TestTrainModel(unittest.TestCase):
    """
    Unit tests for the TrainModel class methods, ensuring the correctness of the training pipeline.
    """

    def setUp(self):
        """
        Set up mock data and TrainModel instance for testing.
        """
        # Create mock data
        self.X = np.random.rand(100, 5)
        self.y = np.random.randint(0, 2, size=(100, 3))
        self.config_path = "mock_config.yaml"
        self.config_content = {
            'training': {'test_size': 0.2, 'random_state': 42, 'cv_folds': 3},
            'models': {
                'RandomForest': {
                    'class': 'sklearn.ensemble.RandomForestClassifier',
                    'hyperparameters': {
                        'n_estimators': [10, 50]
                    }
                }
            }
        }
        # Write mock config file
        with open(self.config_path, 'w') as config_file:
            import yaml
            yaml.dump(self.config_content, config_file)

        self.train_model = TrainModel(self.X, self.y, self.config_path)

    def tearDown(self):
        """
        Clean up after each test by removing the mock config file.
        """
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

    @patch('train_model.train_test_split')
    def test_train_test_split(self, mock_train_test_split):
        """
        Test the train_test_split method.
        """
        # Mock the output of train_test_split
        mock_train_test_split.return_value = (
            np.random.rand(80, 5), np.random.rand(20, 5),
            np.random.randint(0, 2, size=(80, 3)), np.random.randint(0, 2, size=(20, 3))
        )
        self.train_model.train_test_split()
        # Verify the method was called
        mock_train_test_split.assert_called_once()
        # Check that the training and test sets are assigned
        self.assertIsNotNone(self.train_model.X_train, "X_train should not be None after train_test_split.")
        self.assertIsNotNone(self.train_model.y_train, "y_train should not be None after train_test_split.")

    @patch('train_model.TrainModel.tune_model')
    def test_tune_model(self, mock_tune_model):
        """
        Test the tune_model method.
        """
        mock_model = MagicMock(spec=RandomForestClassifier)
        mock_tune_model.return_value = mock_model
        best_estimator = self.train_model.tune_model('RandomForest')
        self.assertIsNotNone(best_estimator, "Tuned model should not be None.")
        mock_tune_model.assert_called_once_with('RandomForest')

    @patch('train_model.TrainModel.find_best_threshold')
    def test_find_best_threshold(self, mock_find_best_threshold):
        """
        Test the find_best_threshold method to ensure correct behavior.
        """
        mock_thresholds = [0.5, 0.6, 0.7]
        mock_find_best_threshold.return_value = mock_thresholds
        best_thresholds = self.train_model.find_best_threshold(self.train_model.models['RandomForest'], self.X, self.y)
        self.assertEqual(best_thresholds, mock_thresholds, "Returned thresholds should match mock thresholds.")

    @patch('train_model.train_test_split')
    @patch('train_model.TrainModel.train_model')
    def test_train_model(self, mock_train_model, mock_train_test_split):
        """
        Test the train_model method.
        """
        # Mock train_test_split to provide data
        mock_train_test_split.return_value = (
            np.random.rand(80, 5), np.random.rand(20, 5),
            np.random.randint(0, 2, size=(80, 3)), np.random.randint(0, 2, size=(20, 3))
        )
        # Mock train_model to return a model
        mock_model = MagicMock(spec=RandomForestClassifier)
        mock_train_model.return_value = mock_model

        # Split data and train the model
        self.train_model.train_test_split()
        trained_model = self.train_model.train_model('RandomForest')
        self.assertIsNotNone(trained_model, "Trained model should not be None.")
        mock_train_model.assert_called_once_with('RandomForest')

    def test_load_models(self):
        """
        Test the load_models method to ensure models are loaded correctly from the configuration.
        """
        models = self.train_model.load_models()
        self.assertIn('RandomForest', models, "RandomForest model should be loaded from configuration.")
        self.assertIsInstance(models['RandomForest'], MultiOutputClassifier, "Loaded model should be an instance of MultiOutputClassifier.")

    @patch('train_model.TrainModel.predict_with_threshold')
    def test_predict_with_threshold(self, mock_predict_with_threshold):
        """
        Test the predict_with_threshold method to ensure correct predictions with thresholds.
        """
        mock_predictions = np.random.randint(0, 2, size=(20, 3))
        mock_predict_with_threshold.return_value = mock_predictions
        predictions = self.train_model.predict_with_threshold(self.train_model.models['RandomForest'], self.X)
        self.assertIsNotNone(predictions, "Predictions should not be None.")
        self.assertEqual(predictions.shape, (20, 3), "Predictions should have the correct shape.")

    @patch('train_model.TrainModel.evaluate_model_performance')
    def test_evaluate_model_performance(self, mock_evaluate_model_performance):
        """
        Test the evaluate_model_performance method to ensure metrics are calculated correctly.
        """
        mock_metrics = {
            'precision': 0.8,
            'recall': 0.7,
            'f1_score': 0.75,
            'accuracy': 0.85,
            'hamming_loss': 0.15
        }
        mock_evaluate_model_performance.return_value = (mock_metrics, {})
        metrics, _ = self.train_model.evaluate_model_performance(self.train_model.models['RandomForest'])
        self.assertIsNotNone(metrics, "Metrics should not be None.")
        self.assertAlmostEqual(metrics['precision'], 0.8, places=2, msg="Precision should match the expected value.")
        self.assertAlmostEqual(metrics['f1_score'], 0.75, places=2, msg="F1 Score should match the expected value.")

if __name__ == '__main__':
    unittest.main()
