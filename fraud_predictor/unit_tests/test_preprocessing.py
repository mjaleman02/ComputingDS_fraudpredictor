import unittest
from unittest.mock import patch
import pandas as pd
from fraud_predictor.preprocessors.preprocessing import load_df, drop_unnecessary_columns

class TestPreprocessingFunctions(unittest.TestCase):

    @patch("fraud_predictor.preprocessors.preprocessing.pd.read_csv")
    def test_load_df(self, mock_read_csv):
        # Create a sample DataFrame to simulate reading from CSV
        sample_data = pd.DataFrame({
            "merchant": ["M1", "M2"],
            "city": ["C1", "C2"],
            "device_fingerprint": ["DF1", "DF2"],
            "ip_address": ["IP1", "IP2"],
            "velocity_last_hour": [10, 20],
            "transaction_id": [1, 2],
            "card_number": ["CN1", "CN2"],
            "merchant_type": ["Type1", "Type2"],
            "some_other_col": [100, 200]
        })

        mock_read_csv.return_value = sample_data
        df = load_df()
        
        # Test that load_df returns a DataFrame and it's not empty
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "Mocked DataFrame should not be empty")

    @patch("fraud_predictor.preprocessors.preprocessing.pd.read_csv")
    def test_drop_unnecessary_columns(self, mock_read_csv):
        # Create a sample DataFrame that includes all columns that need dropping
        sample_data = pd.DataFrame({
            "merchant": ["M1", "M2"],
            "city": ["C1", "C2"],
            "device_fingerprint": ["DF1", "DF2"],
            "ip_address": ["IP1", "IP2"],
            "velocity_last_hour": [10, 20],
            "transaction_id": [1, 2],
            "card_number": ["CN1", "CN2"],
            "merchant_type": ["Type1", "Type2"],
            "some_other_col": [100, 200]
        })

        mock_read_csv.return_value = sample_data
        df = load_df()

        columns_to_check = [
            'merchant', 'city', 'device_fingerprint', 'ip_address',
            'velocity_last_hour', 'transaction_id', 'card_number', 'merchant_type'
        ]

        # Ensure these columns exist before dropping
        for col in columns_to_check:
            self.assertIn(col, df.columns, f"{col} should exist before dropping")

        df_dropped = drop_unnecessary_columns(df)

        # After dropping, these columns should not be present
        for col in columns_to_check:
            self.assertNotIn(col, df_dropped.columns, f"{col} should be dropped")

if __name__ == '__main__':
    unittest.main()
