import os
import pandas as pd

def load_all_data():
    """
    Load all CSV files from the 'data' folder into a dictionary of DataFrames.

    Returns:
        dict: A dictionary where keys are filenames and values are pandas DataFrames.
    """
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    data_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    dataframes = {}

    for file in data_files:
        file_path = os.path.join(data_folder, file)
        try:
            df = pd.read_csv(file_path)
            dataframes[file] = df
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return dataframes
