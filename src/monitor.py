import pandas as pd
import mlflow
import os


def run_monitoring():
    log_file = "logs/prediction_logs.csv"
    alert_file = "logs/alerts.csv"

    if not os.path.exists(log_file):
        print("No prediction logs found.")
        return

    df = pd.read_csv(log_file)
    total_predictions = len(df)
    average_rul = df["predicted_RUL"].mean()
    high_risk_count = int((df["risk_level"] == "High").sum())
    medium_risk_count = int((df["risk_level"] == "Medium").sum())
    low_risk_count = int((df["risk_level"] == "Low").sum())

    alert_count = 0
    if os.path.exists(alert_file):
        alert_count = len(pd.read_csv(alert_file))

    mlflow.set_experiment("Turbofan_RUL_Monitoring")
    with mlflow.start_run(run_name="Prediction_Monitoring_Run"):
        mlflow.log_metric("total_predictions", total_predictions)
        mlflow.log_metric("average_predicted_RUL", average_rul)
        mlflow.log_metric("high_risk_count", high_risk_count)
        mlflow.log_metric("medium_risk_count", medium_risk_count)
        mlflow.log_metric("low_risk_count", low_risk_count)
        mlflow.log_metric("maintenance_alert_count", alert_count)
        mlflow.log_artifact(log_file)
        if os.path.exists(alert_file):
            mlflow.log_artifact(alert_file)

    print("Monitoring completed successfully.")
    print(f"Total predictions: {total_predictions}")
    print(f"Average predicted RUL: {average_rul:.2f}")
    print(f"High risk count: {high_risk_count}")
    print(f"Medium risk count: {medium_risk_count}")
    print(f"Low risk count: {low_risk_count}")
    print(f"Maintenance alert count: {alert_count}")


if __name__ == "__main__":
    run_monitoring()
