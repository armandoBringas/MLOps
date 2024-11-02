import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from mlops.feature_engineering.data_preprocessing import DataPreprocessing
from sklearn.impute import SimpleImputer


class TestDataPreprocessing(unittest.TestCase):
    """
    Unit tests for the DataPreprocessing class methods, ensuring correctness
    of data cleaning, normalization, and PCA processes.
    """

    def setUp(self):
        """
        Set up test data for each test method.
        """
        # Creating a sample DataFrame similar to what the class expects
        self.data = {
            'ID': [1, 2, 3, 4],
            'MV': ['A', 'B', 'C', 'A'],
            'SR': [1.1, 2.2, np.nan, 4.4],
            'NR': [3.1, 4.1, 5.1, 6.1],
            'TR': ['low', 'medium', 'high', 'low'],
            'VR': ['red', 'blue', 'green', 'yellow'],
            'SUR1': [1, 0, 1, 0],
            'SUR2': [1, 2, 1, 3],
            'SUR3': [5, 4, 3, 2],
            'UR': ['X', 'Y', 'Z', 'X'],
            'FR': [10, 15, 20, 25],
            'OR': [3.3, 3.4, 3.5, 3.6],
            'RR': [2.1, 3.1, 4.1, 5.1],
            'BR': [0.1, 0.2, 0.3, 0.4],
            'MR': ['M1', 'M2', 'M3', 'M1'],
            'CR': ['C1', 'C2', 'C3', 'C1'],
            'Green frogs': [0, 1, 0, 1],
            'Brown frogs': [1, 0, 0, 1],
            'Common toad': [0, 0, 1, 1],
            'Fire-bellied toad': [1, 1, 0, 0],
            'Tree frog': [0, 0, 1, 1],
            'Common newt': [1, 0, 0, 1],
            'Great crested newt': [0, 1, 0, 0]
        }
        self.df = pd.DataFrame(self.data)
        self.data_preprocessor = DataPreprocessing(self.df)

    def test_clean_data(self):
        """
        Test the clean_data method for proper cleaning operations.
        """
        cleaned_df = self.data_preprocessor.clean_data()
        self.assertNotIn('ID', cleaned_df.columns, "ID column should be dropped.")
        self.assertEqual(len(cleaned_df), 3, "The first row should be dropped.")
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_df['SR']), "SR should be numeric.")
        self.assertTrue(isinstance(cleaned_df['MV'].dtype, pd.CategoricalDtype), "MV should be categorical.")

    def test_normalize_data(self):
        """
        Test the normalize_data method to ensure data normalization is correct.
        """
        # First, clean the data
        self.data_preprocessor.clean_data()
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        self.data_preprocessor.df[['SR']] = imputer.fit_transform(self.data_preprocessor.df[['SR']])
        normalized_df = self.data_preprocessor.normalize_data()
        self.assertEqual(normalized_df.shape[0], 3, "The number of rows should match cleaned data.")
        self.assertEqual(normalized_df.shape[1], 5, "The number of numerical columns should match.")
        # Check if mean is approximately 0 and variance is approximately 1 for standardized data
        self.assertAlmostEqual(normalized_df.mean().mean(), 0, delta=0.1, msg="Mean of normalized data should be approximately 0.")
        self.assertAlmostEqual(normalized_df.std().mean(), 1, delta=0.3, msg="Standard deviation of normalized data should be approximately 1.")

    def test_apply_pca(self):
        """
        Test the apply_pca method for dimensionality reduction.
        """
        # Clean and normalize data first
        self.data_preprocessor.clean_data()
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        self.data_preprocessor.df[['SR']] = imputer.fit_transform(self.data_preprocessor.df[['SR']])
        normalized_df = self.data_preprocessor.normalize_data()
        # Apply PCA
        principal_components = self.data_preprocessor.apply_pca(normalized_df, variance_retained=0.95)
        # Ensure reduced dimensions are less than original
        self.assertLess(principal_components.shape[1], normalized_df.shape[1], "PCA should reduce the number of features.")

    def test_preprocess_pipeline(self):
        """
        Test the full preprocessing pipeline to ensure the integration of all methods.
        """
        # Handle missing values before running the full pipeline
        imputer = SimpleImputer(strategy='mean')
        self.data_preprocessor.df[['SR']] = imputer.fit_transform(self.data_preprocessor.df[['SR']])
        X_pca, y = self.data_preprocessor.preprocess()
        self.assertEqual(y.shape[1], 7, "Labels should have 7 columns.")
        self.assertLess(X_pca.shape[1], 21, "PCA should have reduced the number of features.")


if __name__ == '__main__':
    unittest.main()
