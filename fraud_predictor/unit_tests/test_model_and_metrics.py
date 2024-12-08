import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted
from fraud_predictor.model.model_and_metrics import (
    split_data, convert_object_to_category, tune_lightgbm,
    train_optimized_lightgbm, predict, predict_proba,
    calculate_accuracy, calculate_roc_auc, calculate_f1,
    plot_feature_importance
)

class TestModelFunctions(unittest.TestCase):

    def setUp(self):
        # Create a synthetic dataset with a categorical and numeric feature, balanced target
        self.df = pd.DataFrame({
            'feature_num': np.arange(1, 31),
            'feature_cat': ['A' if i % 2 == 0 else 'B' for i in range(1, 31)],
            'is_fraud': [0, 1] * 15
        })
        
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            self.df, target_column='is_fraud', test_size=0.5, random_state=42
        )
        
        # Convert object to category
        self.X_train, self.X_test = convert_object_to_category(self.X_train.copy(), self.X_test.copy())

    def test_split_data(self):
        # With test_size=0.5, we expect half the data in train and half in test
        self.assertEqual(len(self.X_train), 15)
        self.assertEqual(len(self.X_test), 15)
        self.assertEqual(len(self.y_train), 15)
        self.assertEqual(len(self.y_test), 15)

    def test_convert_object_to_category(self):
        # Check if feature_cat is converted to category dtype
        self.assertEqual(self.X_train['feature_cat'].dtype.name, 'category')
        self.assertEqual(self.X_test['feature_cat'].dtype.name, 'category')

    @patch("fraud_predictor.model.model_and_metrics.RandomizedSearchCV")
    def test_tune_lightgbm(self, mock_search):
        mock_estimator = MagicMock()
        mock_estimator.best_params_ = {
            'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'max_bin':1500, 'num_leaves':31
        }
        mock_estimator.return_value = mock_estimator
        mock_search.return_value = mock_estimator

        best_params, search = tune_lightgbm(self.X_train, self.y_train, n_iter=2, cv=2, verbose=0)
        self.assertIn('n_estimators', best_params)
        self.assertIn('max_depth', best_params)

    def test_train_optimized_lightgbm(self):
        best_params = {'n_estimators': 10, 'max_depth': 3, 'learning_rate':0.1, 'max_bin':1500, 'num_leaves':31}
        model = train_optimized_lightgbm(self.X_train, self.y_train, best_params)
        check_is_fitted(model, 'feature_importances_')

    def test_predict(self):
        best_params = {'n_estimators':10, 'max_depth':3, 'learning_rate':0.1, 'max_bin':1500, 'num_leaves':31}
        model = train_optimized_lightgbm(self.X_train, self.y_train, best_params)
        preds = predict(model, self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_predict_proba(self):
        best_params = {'n_estimators':10, 'max_depth':3, 'learning_rate':0.1, 'max_bin':1500, 'num_leaves':31}
        model = train_optimized_lightgbm(self.X_train, self.y_train, best_params)
        proba = predict_proba(model, self.X_test)
        self.assertEqual(len(proba), len(self.X_test))
        self.assertTrue((proba >= 0).all() and (proba <= 1).all())

    def test_calculate_accuracy(self):
        best_params = {'n_estimators':10, 'max_depth':3, 'learning_rate':0.1, 'max_bin':1500, 'num_leaves':31}
        model = train_optimized_lightgbm(self.X_train, self.y_train, best_params)
        preds = predict(model, self.X_test)
        acc = calculate_accuracy(self.y_test, preds)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)

    def test_calculate_roc_auc(self):
        best_params = {'n_estimators':10, 'max_depth':3, 'learning_rate':0.1, 'max_bin':1500, 'num_leaves':31}
        model = train_optimized_lightgbm(self.X_train, self.y_train, best_params)
        auc = calculate_roc_auc(model, self.X_test, self.y_test)
        self.assertGreaterEqual(auc, 0)
        self.assertLessEqual(auc, 1)

    def test_calculate_f1(self):
        best_params = {'n_estimators':10, 'max_depth':3, 'learning_rate':0.1, 'max_bin':1500, 'num_leaves':31}
        model = train_optimized_lightgbm(self.X_train, self.y_train, best_params)
        preds = predict(model, self.X_test)
        f1_val = calculate_f1(self.y_test, preds, model_name="test_model")
        self.assertGreaterEqual(f1_val, 0)
        self.assertLessEqual(f1_val, 1)
    @patch("fraud_predictor.model.model_and_metrics.plt.show")
    def test_plot_feature_importance(self, mock_show):
        best_params = {'n_estimators': 10, 'max_depth': 3, 'learning_rate':0.1, 'max_bin':1500, 'num_leaves':31}
        model = train_optimized_lightgbm(self.X_train, self.y_train, best_params)

        # Check if the model produced any non-zero feature importance
        importances = model.feature_importances_
        if all(imp == 0 for imp in importances):
            self.skipTest("No feature importance was generated by the model.")

        # If we have importance values, attempt to plot
        try:
            plot_feature_importance(model, max_features=2, importance_type='gain', title="Test Feature Importance")
        except Exception as e:
            self.fail(f"plot_feature_importance raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
