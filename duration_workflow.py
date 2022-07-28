import pandas as pd

from sklearn.metrics import mean_squared_error
from datetime import timedelta

import xgboost as xgb

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb


@task
def load_data(path):
    data = pd.read_csv(path)
    data['ever_married'] = [0 if i != 'Yes' else 1 for i in data['ever_married']]
    data['gender'] = [0 if i != 'Female' else 1 for i in data['gender']]
    return data


@task(retries=3)
def generate_datasets(df):
    X = df.drop(['stroke'], axis=1)
    y = df['stroke']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)
    return X_train, X_test, y_train, y_test


@task
def train_model(X_train, y_train):
    best_params = {
        "criterion": 'gini',
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0,
        "max_features": None,
        "random_state": None,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0,
        "class_weight": None,
        "ccp_alpha": 0
    }

    clf_gini = DecisionTreeClassifier(best_params)
    clf_gini.fit(X_train, y_train)
    return clf_gini


@task
def estimate_quality(model, X_val, y_val):
    validation = xgb.DMatrix(X_val, label=y_val)
    y_pred = model.predict(validation)
    return mean_squared_error(y_pred, y_val, squared=False)


@flow(task_runner=SequentialTaskRunner())
def nyc_duration_flow():
    df = load_data('data/full_data.csv')
    X_train, X_val, y_train, y_val = generate_datasets(
        df).result()
    model = train_model(X_train, y_train)
    rmse = estimate_quality(model, X_val, y_val)


nyc_duration_flow()
