import pytest
from fraud_predictor.merging.merging_df import add_column_by_merge, load_df2, load_df3

def test_add_column_by_merge():
    """
    Test the add_column_by_merge function to ensure it merges columns correctly.
    """
    # Load the base DataFrame (gdp_country.csv)
    df_base = load_df2('gdp_country.csv')
    
    # Load the DataFrame to merge (gdp_per_capita.csv)
    df_merge = load_df3('gdp_per_capita.csv')
    
    # Ensure data loaded correctly
    assert not df_base.empty, "gdp_country.csv loaded as empty."
    assert not df_merge.empty, "gdp_per_capita.csv loaded as empty."
    
    # Define columns to merge
    merge_on = ['Country Code']  # Adjust based on shared keys
    columns_to_merge = ['2023']  # Column(s) from df_merge to include in the final merged DataFrame
    
    # Add the column by merging
    merged_df = add_column_by_merge(df_base, df_merge, merge_on=merge_on, columns_to_merge=columns_to_merge, how='left')
    
    # Check if both columns with suffixes exist
    assert '2023_x' in merged_df.columns or '2023_y' in merged_df.columns, (
        "Expected columns with suffixes '_x' or '_y' not found in merged DataFrame."
    )
    
    # Validate the merge by checking specific rows
    country_code = 'USA'  # Example country code
    if country_code in df_base['Country Code'].values:
        merged_row = merged_df[merged_df['Country Code'] == country_code]
        if not merged_row.empty:
            base_value = df_base[df_base['Country Code'] == country_code]['Country Name'].values[0]
            merge_value = df_merge[df_merge['Country Code'] == country_code]['2023'].values[0]
            assert merged_row.iloc[0]['Country Name'] == base_value, (
                f"Mismatch in 'Country Name' for {country_code}."
            )
            assert (
                merged_row.iloc[0]['2023_y'] == merge_value
                or merged_row.iloc[0]['2023_x'] == merge_value
            ), f"Mismatch in '2023' value for {country_code}."

