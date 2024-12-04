## import needed packages
import os
import pandas as pd

## The following two functions were used to read the original dataset csv file and create a sample df selecting at random 10000.
## Since the original dataset is too large to upload to github, we only uploaded the sample df and our analysis will start there.
'''           
def load_df():
    """Load the dataset from the given file path."""
    file_path = os.path.join(os.path.dirname(__file__), '../data/synthetic_fraud_data.csv')
    absolute_path = os.path.abspath(file_path)
    return pd.read_csv(absolute_path) 

def sample_data(df, n=10000, random_state=50):
    """
    Randomly select 10000 observations from main_df and return a new pandas df. 
    """
    if n > len(df):
        raise ValueError("Sample size > rows in the DataFrame")
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)
'''

def load_df():
    """Load the dataset from the given file path."""
    file_path = os.path.join(os.path.dirname(__file__), '../data/dropped_df.csv')
    absolute_path = os.path.abspath(file_path)
    return pd.read_csv(absolute_path) 

def drop_unnecessary_columns(df):
    """
    Drop unnecessary columns from a DataFrame and Returns a new pandas df with the specified columns removed.
    """
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
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if not existing_columns_to_drop:
        raise ValueError("The indicated columns don't exits in df")
    
    return df.drop(columns=existing_columns_to_drop)