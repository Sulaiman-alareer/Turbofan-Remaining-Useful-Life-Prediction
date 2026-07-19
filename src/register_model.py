import mlflow
from mlflow.tracking import MlflowClient


def get_best_run(experiment_name, metric_name="RMSE"):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} ASC"],
        max_results=1
    )

    if not runs:
        raise ValueError(f"No runs found in experiment: {experiment_name}")

    return runs[0]


def register_best_model():
    experiment_name = "FD001_RUL_Tuning"
    registered_model_name = "Turbofan_RUL_Predictor"

    client = MlflowClient()
    best_run = get_best_run(experiment_name)

    run_id = best_run.info.run_id
    model_type = best_run.data.params.get("model_name", "Unknown")
    rmse = best_run.data.metrics.get("RMSE")
    mae = best_run.data.metrics.get("MAE")
    r2 = best_run.data.metrics.get("R2")

    model_uri = f"runs:/{run_id}/model"

    print("Best Run Found")
    print(f"Run ID: {run_id}")
    print(f"Model Type: {model_type}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    print(f"Model URI: {model_uri}")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name
    )

    print("\nModel registered successfully")
    print(f"Model Name: {registered_model_name}")
    print(f"Version: {result.version}")

    try:
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="production",
            version=result.version
        )
        print(f"Production alias assigned to version {result.version}")
    except Exception as e:
        print(f"Alias assignment skipped: {e}")


if __name__ == "__main__":
    register_best_model()