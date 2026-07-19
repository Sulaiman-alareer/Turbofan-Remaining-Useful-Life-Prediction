import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def log_sklearn_model(model, x_train):
    input_example = x_train.head(3)
    signature = infer_signature(input_example, model.predict(input_example))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        signature=signature
    )


def tune_random_forest(dataset_name, x_train, y_train, x_test, y_test):
    params_list = [
        {"n_estimators": 50, "max_depth": 8},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 150, "max_depth": 12},
        {"n_estimators": 200, "max_depth": 15},
    ]

    best_rmse = float("inf")
    best_params = None

    for params in params_list:
        with mlflow.start_run(run_name=f"RF_Tuning_{params['n_estimators']}_{params['max_depth']}"):
            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
                n_jobs=-1
            )
            model.fit(x_train, y_train)
            mae, rmse, r2 = evaluate_model(model, x_test, y_test)

            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("model_name", "Random Forest")
            mlflow.log_param("tuning", True)
            mlflow.log_param("rul_clip_value", RUL_CLIP_VALUE)
            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_param("max_depth", params["max_depth"])
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            log_sklearn_model(model, x_train)

            print(f"Random Forest params: {params}")
            print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
            print("-" * 40)

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params

    print(f"Best Random Forest params for {dataset_name}: {best_params}")
    print(f"Best Random Forest RMSE: {best_rmse:.2f}")


def tune_gradient_boosting(dataset_name, x_train, y_train, x_test, y_test):
    params_list = [
        {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 4},
    ]

    best_rmse = float("inf")
    best_params = None

    for params in params_list:
        with mlflow.start_run(run_name=f"GB_Tuning_{params['n_estimators']}_{params['learning_rate']}_{params['max_depth']}"):
            model = GradientBoostingRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                random_state=42
            )
            model.fit(x_train, y_train)
            mae, rmse, r2 = evaluate_model(model, x_test, y_test)

            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("model_name", "Gradient Boosting")
            mlflow.log_param("tuning", True)
            mlflow.log_param("rul_clip_value", RUL_CLIP_VALUE)
            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_param("learning_rate", params["learning_rate"])
            mlflow.log_param("max_depth", params["max_depth"])
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            log_sklearn_model(model, x_train)

            print(f"Gradient Boosting params: {params}")
            print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
            print("-" * 40)

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params

    print(f"Best Gradient Boosting params for {dataset_name}: {best_params}")
    print(f"Best Gradient Boosting RMSE: {best_rmse:.2f}")


def tune_xgboost(dataset_name, x_train, y_train, x_test, y_test):
    if not XGBOOST_AVAILABLE:
        print("XGBoost is not installed. Skipping XGBoost tuning.")
        return

    params_list = [
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 400, "learning_rate": 0.02, "max_depth": 4, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 250, "learning_rate": 0.04, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8},
    ]

    best_rmse = float("inf")
    best_params = None

    for params in params_list:
        run_name = f"XGB_Tuning_{params['n_estimators']}_{params['learning_rate']}_{params['max_depth']}"
        with mlflow.start_run(run_name=run_name):
            model = XGBRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
            model.fit(x_train, y_train)
            mae, rmse, r2 = evaluate_model(model, x_test, y_test)

            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("model_name", "XGBoost")
            mlflow.log_param("tuning", True)
            mlflow.log_param("rul_clip_value", RUL_CLIP_VALUE)
            for key, value in params.items():
                mlflow.log_param(key, value)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            log_sklearn_model(model, x_train)

            print(f"XGBoost params: {params}")
            print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
            print("-" * 40)

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params

    print(f"Best XGBoost params for {dataset_name}: {best_params}")
    print(f"Best XGBoost RMSE: {best_rmse:.2f}")


def run_tuning(dataset_name, train_path, test_path, rul_path):
    print(f"\nStarting tuning for {dataset_name}")
    train_df = load_train_data(train_path)
    test_df = load_test_data(test_path, rul_path)
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)
    mlflow.set_experiment(f"{dataset_name}_RUL_Tuning")
    tune_random_forest(dataset_name, x_train, y_train, x_test, y_test)
    tune_gradient_boosting(dataset_name, x_train, y_train, x_test, y_test)
    tune_xgboost(dataset_name, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    if not XGBOOST_AVAILABLE:
        print("XGBoost is not installed. To enable it, run:")
        print("py -m pip install xgboost")

    run_tuning("FD001", "data/train_FD001.txt", "data/test_FD001.txt", "data/RUL_FD001.txt")
    run_tuning("FD003", "data/train_FD003.txt", "data/test_FD003.txt", "data/RUL_FD003.txt")
