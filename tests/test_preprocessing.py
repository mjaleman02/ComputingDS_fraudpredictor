# tests/test_preprocessing.py

import pytest
from fraud_predictor.data_loader import load_all_data
from fraud_predictor.preprocessors.preprocessing import drop_unnecessary_columns

def test_drop_unnecessary_columns():
    """
    Test the drop_unnecessary_columns function to ensure specified columns are removed.
    """
    # Step 1: Load all data
    data = load_all_data()
    
    # Step 2: Assert that 'dropped_df.csv' is loaded
    assert "dropped_df.csv" in data, "dropped_df.csv not found in loaded data."
    
    # Step 3: Retrieve the DataFrame
    df = data["dropped_df.csv"]
    
    # Step 4: Define columns to drop
    columns_to_drop = [
        'merchant', 
        'city', 
        'device_fingerprint', 
        'ip_address', 
        'velocity_last_hour', 
        'transaction_id', 
        'card_number',
        'merchant_type'
    ]
    
    # Step 5: Assert that columns to drop exist in the DataFrame
    for col in columns_to_drop:
        assert col in df.columns, f"Column '{col}' not found in DataFrame."
    
    # Step 6: Drop the specified columns
    preprocessed_df = drop_unnecessary_columns(df)
    
    # Step 7: Assert that the columns have been dropped
    for col in columns_to_drop:
        assert col not in preprocessed_df.columns, f"Column '{col}' was not dropped."
    
    # Step 8: Assert that other columns remain intact
    remaining_columns = [col for col in df.columns if col not in columns_to_drop]
    for col in remaining_columns:
        assert col in preprocessed_df.columns, f"Column '{col}' should not have been dropped."
    
    # Step 9: Assert that the number of rows remains the same
    assert len(preprocessed_df) == len(df), "Number of rows changed after dropping columns."
