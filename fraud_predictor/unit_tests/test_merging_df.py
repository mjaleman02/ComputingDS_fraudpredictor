import unittest
import pandas as pd
from fraud_predictor.merging.merging_df import add_column_by_merge

class TestMergingFunction(unittest.TestCase):

    def setUp(self):
        # Sample main DataFrame
        self.main_df = pd.DataFrame({
            "country": ["A", "B", "C"],
            "value_main": [10, 20, 30]
        })

        # DataFrame to merge from
        self.merge_df = pd.DataFrame({
            "country": ["A", "B", "C", "D"],
            "value_to_merge": [100, 200, 300, 400]
        })

    def test_add_column_by_merge_left(self):
        # Test with default 'left' join
        merged = add_column_by_merge(
            df=self.main_df,
            df_to_merge=self.merge_df,
            merge_on=["country"],
            columns_to_merge=["value_to_merge"]
        )

        # Check that all rows from main_df are present
        self.assertEqual(len(merged), 3)
        # Check that "value_to_merge" column is present
        self.assertIn("value_to_merge", merged.columns)
        # Check the merged values are correct
        expected_values = [100, 200, 300]
        self.assertListEqual(merged["value_to_merge"].tolist(), expected_values)

    def test_add_column_by_merge_inner(self):
        # Test with an 'inner' join
        merged = add_column_by_merge(
            df=self.main_df,
            df_to_merge=self.merge_df,
            merge_on=["country"],
            columns_to_merge=["value_to_merge"],
            how="inner"
        )

        # For inner join, only matching rows remain: 'A', 'B', 'C' match; 'D' is excluded
        self.assertEqual(len(merged), 3)
        self.assertIn("value_to_merge", merged.columns)
        expected_values = [100, 200, 300]
        self.assertListEqual(merged["value_to_merge"].tolist(), expected_values)

    def test_add_column_by_merge_right(self):
        # Test a 'right' join to ensure that rows in df_to_merge not in main_df still appear
        merged = add_column_by_merge(
            df=self.main_df,
            df_to_merge=self.merge_df,
            merge_on=["country"],
            columns_to_merge=["value_to_merge"],
            how="right"
        )

        # For right join, we get 'A', 'B', 'C' plus 'D' from merge_df
        self.assertEqual(len(merged), 4)
        self.assertIn("value_to_merge", merged.columns)

        # Since 'D' does not appear in main_df, 'value_main' should be NaN for that row
        self.assertTrue(pd.isna(merged.loc[merged["country"] == "D", "value_main"]).all())

    def test_add_column_by_merge_no_common_values(self):
        # Test scenario where main_df and merge_df share no keys
        # Overwriting main_df to have distinct keys
        main_df_no_common = pd.DataFrame({
            "country": ["X", "Y", "Z"],
            "value_main": [40, 50, 60]
        })

        merged = add_column_by_merge(
            df=main_df_no_common,
            df_to_merge=self.merge_df,
            merge_on=["country"],
            columns_to_merge=["value_to_merge"],
            how="left"
        )

        # No common keys means merged columns are all NaN
        self.assertEqual(len(merged), 3)
        self.assertIn("value_to_merge", merged.columns)
        self.assertTrue(merged["value_to_merge"].isna().all())

if __name__ == '__main__':
    unittest.main()
