# tests/test_model_and_metrics.py

import pytest
from fraud_predictor.data_loader import load_all_data
from fraud_predictor.model import (
    split_data,
    convert_object_to_category,
    tune_lightgbm,
    train_optimized_lightgbm,
    predict,
    predict_proba,
    calculate_accuracy,
    calculate_roc_auc,
    calculate_f1,
    plot_feature_importance
)

def test_model_training():
    """
    Test the complete model training pipeline:
    1. Split data
    2. Convert object columns to categories
    3. Tune LightGBM hyperparameters
    4. Train optimized LightGBM model
    5. Make predictions
    6. Calculate metrics
    """
    # Step 1: Load all data
    data = load_all_data()
    
    # Assert that 'dropped_df.csv' is loaded
    assert "dropped_df.csv" in data, "dropped_df.csv not found in loaded data."
    
    # Step 2: Retrieve the DataFrame
    df = data["dropped_df.csv"]
    
    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df, target_column='is_fraud', test_size=0.2, random_state=50)
    
    # Step 4: Convert object columns to categorical
    X_train_cat, X_test_cat = convert_object_to_category(X_train.copy(), X_test.copy())
    
    # Step 5: Tune LightGBM hyperparameters
    best_params, random_search = tune_lightgbm(X_train_cat, y_train, n_iter=10, cv=3, random_state=1, verbose=0, n_jobs=-1)
    
    # Assert that best_params are returned
    assert isinstance(best_params, dict), "Best parameters not returned as a dictionary."
    assert random_search is not None, "RandomizedSearchCV object not returned."
    
    # Step 6: Train optimized LightGBM model
    optimized_model = train_optimized_lightgbm(X_train_cat, y_train, best_params)
    
    # Assert that the model is trained
    assert optimized_model is not None, "Optimized model training failed, returned None."
    assert hasattr(optimized_model, "predict"), "Optimized model does not have a 'predict' method."
    
    # Step 7: Make predictions
    y_pred = predict(optimized_model, X_test_cat)
    y_proba = predict_proba(optimized_model, X_test_cat)
    
    # Step 8: Calculate metrics
    accuracy = calculate_accuracy(y_test, y_pred)
    roc_auc = calculate_roc_auc(optimized_model, X_test_cat, y_test)
    f1 = calculate_f1(y_test, y_pred, model_name="Optimized LightGBM")
    
    # Step 9: Assertions on metrics
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1."
    assert 0 <= roc_auc <= 1, "ROC AUC should be between 0 and 1."
    assert 0 <= f1 <= 1, "F1 score should be between 0 and 1."
    
