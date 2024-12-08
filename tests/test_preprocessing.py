import pytest
from fraud_predictor.data_loader import load_all_data
from fraud_predictor.preprocessors.preprocessing import drop_unnecessary_columns

def test_drop_unnecessary_columns():
    """
    Test the drop_unnecessary_columns function to ensure specified columns are removed.
    """
    # Load data
    data = load_all_data()
    
    # Assert that 'dropped_df.csv' is loaded
    assert "dropped_df.csv" in data, "dropped_df.csv not found in loaded data."
    
    # Define df
    df = data["dropped_df.csv"]
    
    # Define columns to drop
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
    
    # Assert that columns to drop exist in the DataFrame
    for col in columns_to_drop:
        assert col in df.columns, f"Column '{col}' not found in DataFrame."
    
    # Drop the specified columns
    preprocessed_df = drop_unnecessary_columns(df)
    
    # Assert that the specified columns have been dropped
    for col in columns_to_drop:
        assert col not in preprocessed_df.columns, f"Column '{col}' was not dropped."
    
    # Check if other columns remained intact
    remaining_columns = [col for col in df.columns if col not in columns_to_drop]
    for col in remaining_columns:
        assert col in preprocessed_df.columns, f"Column '{col}' should not have been dropped."
    
    # Verify if the number of rows remained the same
    assert len(preprocessed_df) == len(df), "Number of rows changed after dropping columns."
