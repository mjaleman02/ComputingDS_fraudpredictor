## import needed packages 
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt
from scipy.stats import uniform
    
def split_data(df, target_column='is_fraud', test_size=0.2, random_state=50):
    """
    Split between train and test, ensuring balance between classes in the target column. 
    """
    X = df.drop(columns=target_column)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def convert_object_to_category(X_train, X_test):
    """
    Convert columns of type 'object' in X_train and X_test into pandas Categoricals.
    """
    categorical_columns = X_train.select_dtypes(include='object').columns
    for col in categorical_columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')
    
    return X_train, X_test


def tune_lightgbm(X_train, y_train, n_iter=50, cv=5, random_state=1, verbose=2, n_jobs=-1):
    """
    Perform hyperparameter tuning for a LightGBM classifier using Random Search.
    """
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': uniform(0.01, 0.2),
        'max_bin': [1500, 2000],
        'num_leaves': [31, 50, 100]
    }
    random_search = RandomizedSearchCV(
        estimator=lgb.LGBMClassifier(random_state=random_state),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_jobs
    )
    random_search.fit(X_train, y_train)
    return random_search.best_params_, random_search


def train_optimized_lightgbm(X_train, y_train, best_params):
    """
    Train an optimized LightGBM model using the best hyperparameters.
    
    best_params: The best hyperparameters obtained from tuning.
    """
    optimized_model = lgb.LGBMClassifier(**best_params, random_state=1)
    optimized_model.fit(X_train, y_train)
    return optimized_model


def predict(model, X_test):
    """Make predictions on the test set."""
    return model.predict(X_test)


def predict_proba(model, X):
    """Predict probabilities with the trained model."""
    return model.predict_proba(X)[:, 1]


def calculate_accuracy(y_test, y_pred):
    """Compute the accuracy of the model."""
    return accuracy_score(y_test, y_pred)


def calculate_roc_auc(model, X, y):
    """Compute ROC_AUC metric using predicted probabilities."""
    y_proba = model.predict_proba(X)[:, 1]  
    return roc_auc_score(y, y_proba)


def calculate_f1(y_test, predictions, model_name="model", average='weighted'):
    """Compute the F1 score for a given model's predictions."""
    f1 = f1_score(y_test, predictions, average=average)
    print(f"{model_name} F1 score: {f1:.4f}")
    return f1


def plot_feature_importance(model, max_features=10, importance_type='gain', title="Feature Importance"):
    """ Plot the feature importance of a LightGBM model."""
    ax = lgb.plot_importance(model, max_num_features=max_features, importance_type=importance_type)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    plt.show()
        