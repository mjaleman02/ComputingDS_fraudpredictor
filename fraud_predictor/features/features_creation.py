## import needed packages
import pandas as pd
        
def transform_to_datetime_type(df):
    """
    Transform the 'timestamp' column of a DataFrame to datetime format.
    """
    if 'timestamp' not in df.columns:
        raise ValueError("The column 'timestamp' does not exist in the DataFrame.")
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        raise ValueError(f"Failed to convert the 'timestamp' column to datetime: {e}")
    return df


def create_time_columns(df):
    """
    Create columns for month, day, and hour from the 'timestamp' column.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        raise ValueError("The 'timestamp' column must be in datetime format.")
    df['transaction_month'] = df['timestamp'].dt.month
    df['transaction_day'] = df['timestamp'].dt.day
    df['transaction_hour'] = df['timestamp'].dt.hour
    
    return df


def categorize_hour_column(df):
    """
    Categorize the 'transaction_hour' column into time-of-day categories.
    """
    if 'transaction_hour' not in df.columns:
        raise ValueError("The column 'transaction_hour' does not exist in the DataFrame.")
    time_bins = [0, 6, 12, 15, 19, 21, 24]
    time_labels = ['morning', 'lunch', 'afternoon', 'dinner', 'evening', 'night']
    df['hour_category'] = pd.cut(df['transaction_hour'], bins=time_bins, labels=time_labels, right=False)
    
    return df


def drop_redundant_columns(df, columns_to_drop):
    """
    Drop columns that became redundant and/or unnecessary after the variable transformations performed.
    
    - columns_to_drop (list): A list of column names to drop.
    """
    return df.drop(columns=columns_to_drop)


def create_channel_usage(df, customer_col='customer_id', channel_col='channel'):
    """
    Calculate the frequency with which each customer makes purchases using each channel.
    """
    # Count how many times each channel is used by each customer
    channel_count = df.groupby([customer_col, channel_col])[channel_col].transform('count')
    
    # Calculate the total transactions per customer
    total_transactions = df.groupby(customer_col)[channel_col].transform('count')

    df['channel_usage'] = channel_count / total_transactions
    
    return df


def create_interaction_by_category(df, col1, col2, new_col_name):
    """
    Create an interaction term by normalizing a numeric column (col1) within each category (col2).
    
    - col1 : numeric column to normalize.
    - col2 : category column used for grouping.
    """
    # Calculate the mean of col1 per col2
    category_mean = df.groupby(col2)[col1].transform('mean')
    
    # Create the new interaction term column
    df[new_col_name] = df[col1] / category_mean
    
    return df


def create_payment_safety(df, device_col='device', safety_col='payment_safety', device_mapping=None):
    """
    Map the values of the 'device' column to payment safety levels and add the result
    as a new column. The mapping is provided at function call for flexibility.

    - device_col : The column containing payment methods.
    - safety_col : The name of the new column to store the safety levels.
    - device_mapping : A dictionary mapping device/payment methods to safety levels.
    """
    if device_mapping is None:
        raise ValueError("You must provide a device_mapping dictionary.")
    
    df[safety_col] = df[device_col].map(device_mapping)
    return df