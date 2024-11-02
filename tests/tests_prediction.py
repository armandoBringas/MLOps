import unittest
import numpy as np
import pickle
import os
import sys

# Add the parent directory to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlops', 'modeling')))

from prediction import Predictor
from unittest.mock import patch
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class TestPredictor(unittest.TestCase):
    """
    Unit tests for the Predictor class methods, ensuring the correctness of the prediction pipeline.
    """

    def setUp(self):
        """
        Set up mock data and Predictor instance for testing.
        """
        # Create mock data
        self.X = np.random.rand(20, 5)
        self.model_path = "mock_model.pkl"

        # Create a simple RandomForest model
        mock_model = RandomForestClassifier()
        mock_model.fit(np.random.rand(50, 5), np.random.randint(0, 2, size=(50, 3)))

        # Save the mock model
        with open(self.model_path, 'wb') as model_file:
            pickle.dump(mock_model, model_file)

        self.predictor = Predictor(self.model_path, pca_path=None)
        # Load all components to make sure they are properly initialized
        self.predictor.load_all()

    def tearDown(self):
        """
        Clean up after each test by removing the mock model file.
        """
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_load_model(self):
        """
        Test that the model is loaded correctly.
        """
        self.assertIsNotNone(self.predictor.model, "Model should be loaded and not None.")

    def test_preprocess_input(self):
        """
        Test the preprocess_input method to ensure input data is correctly processed.
        """
        # Create a DataFrame to test preprocessing
        df_input = pd.DataFrame(self.X, columns=[f"feature_{i}" for i in range(self.X.shape[1])])
        processed_data = self.predictor.preprocess_input(df_input)
        self.assertIsNotNone(processed_data, "Processed data should not be None.")
        self.assertEqual(processed_data.shape, self.X.shape, "Processed data should have the same shape as input data.")

    def test_predict_with_thresholds(self):
        """
        Test the predict method to ensure predictions are thresholded correctly.
        """
        # Mock predict_proba to return probabilities that match the number of samples in self.X
        num_samples = self.X.shape[0]
        num_classes = 3
        mock_probabilities = np.random.rand(num_samples, num_classes)
        self.predictor.model.predict_proba = lambda X: [mock_probabilities[:, i] for i in range(mock_probabilities.shape[1])]

        # Perform prediction
        predictions = self.predictor.predict(self.X)

        # Check the shape of the predictions
        self.assertEqual(predictions.shape, (num_samples, num_classes), "Predictions should have the correct shape.")
        
        # Check that the predictions are thresholded correctly
        expected_predictions = (mock_probabilities > 0.5).astype(int)
        self.assertTrue((predictions == expected_predictions).all(), "Predictions should be thresholded correctly.")

    def test_decode_prediction(self):
        """
        Test the decode_prediction method to ensure predictions are decoded correctly.
        """
        # Mock predictions and mock the labels for the model
        mock_prediction = np.array([[1, 0, 1], [0, 1, 0]])
        self.predictor.class_labels = ['Green frogs', 'Common toad', 'Brown frogs']
        decoded_predictions = self.predictor.decode_prediction(mock_prediction)

        # Check that the decoded predictions are correctly formatted
        expected_decoded = [['Green frogs', 'Brown frogs'], ['Common toad']]
        self.assertEqual(decoded_predictions, expected_decoded, "Decoded predictions should match the expected labels.")

if __name__ == '__main__':
    unittest.main()
