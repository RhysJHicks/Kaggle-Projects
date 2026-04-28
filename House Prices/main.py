import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# ----------------------------
# 1. AUTO DATA LOADER
# ----------------------------
class DataLoader:
    def __init__(self, project_folder):
        self.base = Path(project_folder)

    def load(self):
        files = list(self.base.glob("*.csv"))

        train = pd.read_csv([f for f in files if "train" in f.name][0])
        test = pd.read_csv([f for f in files if "test" in f.name][0])

        return train, test


# ----------------------------
# 2. PREPROCESSOR
# ----------------------------
def preprocess(X):
    X = X.fillna(-999)
    X = pd.get_dummies(X)
    return X


# ----------------------------
# 3. METRICS
# ----------------------------
def get_metric(name):
    if name == "rmse":
        return lambda y, p: mean_squared_error(y, p, squared=False)
    elif name == "mae":
        return mean_absolute_error
    else:
        raise ValueError("Unknown metric")


# ----------------------------
# 4. MODELS
# ----------------------------
def get_models():
    return {
        "xgboost": XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        "catboost": CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_seed=42
        )
    }


# ----------------------------
# 5. ENGINE
# ----------------------------
class KaggleEngine:

    def run(self, project_folder, target, metric_name):

        # Load data automatically
        loader = DataLoader(project_folder)
        train, test = loader.load()

        print("\n📦 Data Loaded")

        X = train.drop(target, axis=1)
        y = train[target]

        # log transform (safe for house prices style problems)
        y = np.log1p(y)

        # split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # preprocess
        X_train = preprocess(X_train)
        X_val = preprocess(X_val)

        # align columns
        X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)

        metric = get_metric(metric_name)
        models = get_models()

        results = {}

        print("\n🚀 Training Models...\n")

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            rmse = metric(y_val, preds)
            results[name] = rmse

            print(f"{name.upper()} → {metric_name}: {rmse:.5f}")

        best_model = min(results, key=results.get)

        print("\n🏆 BEST MODEL:", best_model)

        return models[best_model], X_train, X_val, y_train, y_val, test


# ----------------------------
# 6. MENU INTERFACE
# ----------------------------
def menu():

    print("\n========================")
    print("KAGGLE AUTO ENGINE")
    print("========================")

    project = input("Enter dataset folder path: ")
    target = input("Enter target column name: ")
    metric = input("Metric (rmse / mae): ")

    engine = KaggleEngine()
    model, X_train, X_val, y_train, y_val, test = engine.run(
        project, target, metric
    )

    print("\n✅ Training complete!")
    print("Best model selected and ready for predictions.")


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    menu()
mport kagglehub

# Download latest version
path = kagglehub.competition_download('house-prices-advanced-regression-techniques')

print("Path to competition files:", path)