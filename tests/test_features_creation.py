# tests/test_features_creation.py

import pytest
from fraud_predictor.data_loader import load_all_data
from fraud_predictor.features.features_creation import create_features

def test_create_features():
    """
    Test the create_features function to ensure all features are created correctly.
    """
    # Step 1: Load all data
    data = load_all_data()
    
    # Step 2: Assert that 'dropped_df.csv' is loaded
    assert "dropped_df.csv" in data, "dropped_df.csv not found in loaded data."
    
    # Step 3: Retrieve the DataFrame
    df = data["dropped_df.csv"]
    
    # Step 4: Apply the create_features function
    feature_df = create_features(df)
    
    # Step 5: Define expected new columns
    expected_new_columns = [
        'transaction_month',
        'transaction_day',
        'hour_category',
        'channel_usage',
        'channel_hour_interaction',
        'payment_safety'
    ]
    
    # Step 6: Assert that new columns are present
    for col in expected_new_columns:
        assert col in feature_df.columns, f"Expected feature column '{col}' not found."
    
    # Step 7: Assert that columns to drop are no longer present
    columns_to_drop = ['timestamp', 'transaction_hour']  # Adjust based on actual columns dropped
    for col in columns_to_drop:
        assert col not in feature_df.columns, f"Column '{col}' was not dropped."
    
    # Step 8: Assert that existing columns remain intact
    existing_columns = [col for col in df.columns if col not in columns_to_drop]
    for col in existing_columns:
        assert col in feature_df.columns, f"Existing column '{col}' should not have been dropped."
    
    # Step 9: Assert that the number of rows remains unchanged
    assert len(feature_df) == len(df), "Number of rows changed after feature creation."
    
    