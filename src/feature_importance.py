import os
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt

MODEL_NAME = "Turbofan_RUL_Predictor"
MODEL_ALIAS = "production"

FEATURE_COLUMNS = (
    ["cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def get_model_uri():
    client = MlflowClient()
    try:
        client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
        return f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    except Exception:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            raise ValueError(f"No registered versions found for model: {MODEL_NAME}")
        latest_version = max(versions, key=lambda version: int(version.version))
        return f"models:/{MODEL_NAME}/{latest_version.version}"


def get_sklearn_model(pyfunc_model):
    sklearn_model = pyfunc_model._model_impl.sklearn_model
    if hasattr(sklearn_model, "feature_importances_"):
        return sklearn_model
    if hasattr(sklearn_model, "named_steps"):
        for step in sklearn_model.named_steps.values():
            if hasattr(step, "feature_importances_"):
                return step
    return sklearn_model


def extract_feature_importance():
    model_uri = get_model_uri()
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    sklearn_model = get_sklearn_model(pyfunc_model)

    if not hasattr(sklearn_model, "feature_importances_"):
        print("This model does not support feature importance.")
        print("Feature importance works with Random Forest, Gradient Boosting, or XGBoost.")
        return

    importances = sklearn_model.feature_importances_
    if len(importances) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Feature count mismatch. Model has {len(importances)} importances, "
            f"but FEATURE_COLUMNS has {len(FEATURE_COLUMNS)} columns."
        )

    importance_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances})
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    os.makedirs("reports", exist_ok=True)
    csv_path = "reports/feature_importance.csv"
    plot_path = "reports/feature_importance.png"
    importance_df.to_csv(csv_path, index=False)

    top_features = importance_df.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print("Feature importance saved successfully.")
    print(f"Model URI: {model_uri}")
    print(f"CSV file: {csv_path}")
    print(f"Plot file: {plot_path}")
    print("\nTop 10 features:")
    print(top_features)


if __name__ == "__main__":
    extract_feature_importance()
