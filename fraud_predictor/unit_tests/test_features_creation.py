import unittest
import pandas as pd
from fraud_predictor.features.features_creation import (
    transform_to_datetime_type,
    create_time_columns,
    categorize_hour_column,
    drop_redundant_columns,
    create_channel_usage,
    create_interaction_by_category,
    create_payment_safety,
    create_features
)

class TestFeatureCreation(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame that includes all necessary columns
        self.df = pd.DataFrame({
            'timestamp': [
                '2023-01-01 00:30:00',
                '2023-01-01 12:00:00',
                '2023-01-01 18:45:00'
            ],
            'customer_id': [101, 101, 102],
            'channel': ['web', 'mobile', 'web'],
            'device': ['web', 'mobile', 'POS']
        })

    def test_transform_to_datetime_type_success(self):
        df_transformed = transform_to_datetime_type(self.df.copy())
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_transformed['timestamp']))

    def test_transform_to_datetime_type_no_timestamp_column(self):
        df_no_ts = self.df.drop(columns=['timestamp'])
        with self.assertRaises(ValueError):
            transform_to_datetime_type(df_no_ts)

    def test_create_time_columns(self):
        df_dt = transform_to_datetime_type(self.df.copy())
        df_time = create_time_columns(df_dt)
        self.assertIn('transaction_month', df_time.columns)
        self.assertIn('transaction_day', df_time.columns)
        self.assertIn('transaction_hour', df_time.columns)
        # Check correct type (integers for these derived columns)
        self.assertTrue(pd.api.types.is_integer_dtype(df_time['transaction_month']))
        self.assertTrue(pd.api.types.is_integer_dtype(df_time['transaction_day']))
        self.assertTrue(pd.api.types.is_integer_dtype(df_time['transaction_hour']))

    def test_create_time_columns_non_datetime(self):
        with self.assertRaises(ValueError):
            # original df is not converted to datetime
            create_time_columns(self.df.copy())

    def test_categorize_hour_column(self):
        # First create the hour column
        df_dt = transform_to_datetime_type(self.df.copy())
        df_time = create_time_columns(df_dt)
        df_cat = categorize_hour_column(df_time)
        self.assertIn('hour_category', df_cat.columns)
        self.assertEqual(df_cat['hour_category'].dtype.name, 'category')

    def test_categorize_hour_column_missing(self):
        # Without transaction_hour column
        with self.assertRaises(ValueError):
            categorize_hour_column(self.df.copy())

    def test_drop_redundant_columns(self):
        df_dt = transform_to_datetime_type(self.df.copy())
        df_time = create_time_columns(df_dt)
        columns_to_drop = ['timestamp', 'transaction_hour']
        df_dropped = drop_redundant_columns(df_time, columns_to_drop)
        for col in columns_to_drop:
            self.assertNotIn(col, df_dropped.columns)

    def test_create_channel_usage(self):
        # Need channel and customer_id columns
        df_dt = transform_to_datetime_type(self.df.copy())
        df_time = create_time_columns(df_dt)
        df_cat = categorize_hour_column(df_time)
        # Just test this on the prepared df
        df_usage = create_channel_usage(df_cat, customer_col='customer_id', channel_col='channel')
        self.assertIn('channel_usage', df_usage.columns)
        # Check that channel_usage is between 0 and 1
        self.assertTrue((df_usage['channel_usage'] >= 0).all() and (df_usage['channel_usage'] <= 1).all())

    def test_create_interaction_by_category(self):
        # We need 'channel_usage' and 'hour_category'
        df_dt = transform_to_datetime_type(self.df.copy())
        df_time = create_time_columns(df_dt)
        df_cat = categorize_hour_column(df_time)
        df_usage = create_channel_usage(df_cat, customer_col='customer_id', channel_col='channel')
        df_interaction = create_interaction_by_category(
            df_usage,
            col1='channel_usage',
            col2='hour_category',
            new_col_name='channel_hour_interaction'
        )
        self.assertIn('channel_hour_interaction', df_interaction.columns)
        # Check that the interaction is a ratio around 1 on average
        # Not a strict requirement, but channel_usage mean per category should be about 1.
        # This is just a sanity check.
        self.assertFalse(df_interaction['channel_hour_interaction'].isna().any())

    def test_create_payment_safety(self):
        # We need 'device' column
        df_safety = create_payment_safety(
            self.df.copy(),
            device_col='device', 
            safety_col='payment_safety',
            device_mapping={'mobile': 'high', 'web': 'medium', 'POS': 'low'}
        )
        self.assertIn('payment_safety', df_safety.columns)
        expected = ['medium', 'high', 'low']
        self.assertListEqual(df_safety['payment_safety'].tolist(), expected)

    def test_create_payment_safety_no_mapping(self):
        with self.assertRaises(ValueError):
            create_payment_safety(self.df.copy(), device_col='device', safety_col='payment_safety', device_mapping=None)

    def test_create_features(self):
        # End-to-end test
        df_features = create_features(self.df.copy())
        # Check all expected columns after create_features
        expected_cols = [
            'customer_id', 'channel', 'device', 'transaction_month',
            'transaction_day', 'hour_category', 'channel_usage',
            'channel_hour_interaction', 'payment_safety'
        ]
        for col in expected_cols:
            self.assertIn(col, df_features.columns)

        # Check that timestamp and transaction_hour are dropped
        self.assertNotIn('timestamp', df_features.columns)
        self.assertNotIn('transaction_hour', df_features.columns)

if __name__ == '__main__':
    unittest.main()
