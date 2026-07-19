import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature

from preprocess import load_train_data, load_test_data, split_features_target, RUL_CLIP_VALUE

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return mae, rmse, r2


def get_models():
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

    return models


def log_model_params(model_name):
    if model_name == "Random Forest":
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 12)
    elif model_name == "Gradient Boosting":
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 3)
    elif model_name == "XGBoost":
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("learning_rate", 0.03)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("subsample", 0.8)
        mlflow.log_param("colsample_bytree", 0.8)


def train_and_log(dataset_name, train_path, test_path, rul_path):
    print(f"\nTraining models for {dataset_name}")

    train_df = load_train_data(train_path)
    test_df = load_test_data(test_path, rul_path)

    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    models = get_models()
    mlflow.set_experiment(f"{dataset_name}_RUL_Prediction")

    best_rmse = float("inf")
    best_model_name = ""
    input_example = x_train.head(3)

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(x_train, y_train)
            mae, rmse, r2 = evaluate_model(model, x_test, y_test)
            signature = infer_signature(input_example, model.predict(input_example))

            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("problem_type", "Regression")
            mlflow.log_param("target", "Remaining Useful Life")
            mlflow.log_param("rul_clip_value", RUL_CLIP_VALUE)
            mlflow.log_param("xgboost_available", XGBOOST_AVAILABLE)
            log_model_params(model_name)

            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                signature=signature
            )

            print(f"{model_name}")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"R2: {r2:.4f}")
            print("-" * 30)

            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = model_name

    print(f"Best model for {dataset_name}: {best_model_name}")
    print(f"Best RMSE: {best_rmse:.2f}")
    return best_model_name, best_rmse


if __name__ == "__main__":
    if not XGBOOST_AVAILABLE:
        print("XGBoost is not installed. Training will continue without XGBoost.")
        print("To enable it, run: py -m pip install xgboost")

    train_and_log("FD001", "data/train_FD001.txt", "data/test_FD001.txt", "data/RUL_FD001.txt")
    train_and_log("FD003", "data/train_FD003.txt", "data/test_FD003.txt", "data/RUL_FD003.txt")
