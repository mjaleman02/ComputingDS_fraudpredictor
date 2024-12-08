import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform
from matplotlib import pyplot as plt

def split_data(df, target_column='is_fraud', test_size=0.2, random_state=50):
    X = df.drop(columns=target_column)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def convert_object_to_category(X_train, X_test):
    categorical_columns = X_train.select_dtypes(include='object').columns
    for col in categorical_columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')
    return X_train, X_test


def tune_lightgbm(X_train, y_train, n_iter=50, cv=5, random_state=1, verbose=2, n_jobs=-1):
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
    optimized_model = lgb.LGBMClassifier(**best_params, random_state=1)
    optimized_model.fit(X_train, y_train)
    return optimized_model


def predict(model, X_test):
    return model.predict(X_test)


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]


def calculate_accuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred)


def calculate_roc_auc(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_proba)


def calculate_f1(y_test, predictions, model_name="model", average='weighted'):
    f1 = f1_score(y_test, predictions, average=average)
    print(f"{model_name} F1 score: {f1:.4f}")
    return f1


def plot_feature_importance(model, max_features=10, importance_type='gain', title="Feature Importance"):
    ax = lgb.plot_importance(model, max_num_features=max_features, importance_type=importance_type)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    plt.show()
