## import needed packages
import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt
from scipy.stats import uniform
        
## Functions associated with GDP dataset ##
def load_df2(file_name):
    """
    Load a CSV file, specifying the path when calling the function.
    """
    file_path = os.path.join(os.path.dirname(__file__), '../data/' + file_name)
    absolute_path = os.path.abspath(file_path)
    return pd.read_csv(absolute_path) 


def rename_columns(df, column_mapping):
    """
    Apply a mapping to rename columns in the DataFrame.
    
    - column_mapping : A dictionary mapping old names to new names, where:
        - Keys are the old column names or values to map.
        - Values are the new column names or values.
    """
    df = df.rename(columns=column_mapping)
    return df


def rename_values(df, column_mapping):
    """
    Apply a mapping to map values in specific columns of a DataFrame.
    
    - column_mapping : A dictionary mapping old names to new names, where:
        - Keys are the old column names or values to map.
        - Values are the new column names or values.
    """
    for col, mapping in column_mapping.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)  # Map values within a column
        else:
            df = df.rename(columns={col: mapping})  # Rename a column
    return df


def add_column_by_merge(df, df_to_merge, merge_on, columns_to_merge, how='left'):
    """
    Add specific columns to the main DataFrame by merging it with another DataFrame.

    - df_to_merge : The DataFrame to merge from.
    - merge_on : The column(s) to merge on.
    - columns_to_merge : The columns to include from the second DataFrame.
    """
    df = df.merge(df_to_merge[merge_on + columns_to_merge], on=merge_on, how=how)
    return df


## Functions associated with GDP per capita datset ##
def load_df3(file_name):
    """
    Load a CSV file, specifying the path when calling the function.
    """
    file_path = os.path.join(os.path.dirname(__file__), '../data/' + file_name)
    absolute_path = os.path.abspath(file_path) 
    
    return pd.read_csv(absolute_path, header=2)


def gdp_capita_columns_keep(df):
    """
    Keep only necessary columns from a DataFrame and return a new pandas df with the specified columns only.
    """
    columns_to_keep = [
        "Country Name", 
        "Country Code",
        "Indicator Code",
        "2023"
    ]
    existing_columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    
    if not existing_columns_to_keep:
        raise ValueError("The indicated columns don't exits in df")
    
    return df[existing_columns_to_keep]