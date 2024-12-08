import pytest
from fraud_predictor.data_loader import load_all_data
from fraud_predictor.features.features_creation import create_features

def test_create_features():
    """
    Test the create_features function to ensure all features are created correctly.
    """
    # Load data
    data = load_all_data()
    
    # Assert that 'dropped_df.csv' is loaded
    assert "dropped_df.csv" in data, "dropped_df.csv not found in loaded data."
    
    # Define df
    df = data["dropped_df.csv"]
    
    # Apply the create_features function
    feature_df = create_features(df)
    
    # Define expected new columns
    expected_new_columns = [
        'transaction_month',
        'transaction_day',
        'hour_category',
        'channel_usage',
        'channel_hour_interaction',
        'payment_safety'
    ]
    
    # Assert that the columns defined above are present
    for col in expected_new_columns:
        assert col in feature_df.columns, f"Expected feature column '{col}' not found."
    
    # Verify that the columns to drop are no longer present
    columns_to_drop = ['timestamp', 'transaction_hour']  
    for col in columns_to_drop:
        assert col not in feature_df.columns, f"Column '{col}' was not dropped."
    
    # Check whether other columns remained intact
    existing_columns = [col for col in df.columns if col not in columns_to_drop]
    for col in existing_columns:
        assert col in feature_df.columns, f"Existing column '{col}' should not have been dropped."
    
    # Check if the number of rows involved in the feature creation matches with expectations
    assert len(feature_df) == len(df), "Number of rows changed after feature creation."
    
    