from flask import Flask, request, jsonify
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

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
        model_version = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
        return f"models:/{MODEL_NAME}@{MODEL_ALIAS}", model_version.version
    except Exception:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            raise ValueError(f"No registered versions found for model: {MODEL_NAME}")
        latest_version = max(versions, key=lambda version: int(version.version))
        return f"models:/{MODEL_NAME}/{latest_version.version}", latest_version.version


MODEL_URI, MODEL_VERSION = get_model_uri()
model = mlflow.pyfunc.load_model(MODEL_URI)


def get_risk_level(rul):
    if rul <= 30:
        return "High"
    elif rul <= 80:
        return "Medium"
    return "Low"


def log_prediction(input_data, predicted_rul, risk_level):
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/prediction_logs.csv"

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "predicted_RUL": predicted_rul,
        "risk_level": risk_level
    }
    row.update(input_data)
    df = pd.DataFrame([row])

    if not os.path.exists(log_file):
        df.to_csv(log_file, index=False)
    else:
        df.to_csv(log_file, mode="a", header=False, index=False)


def log_alert(input_data, predicted_rul, risk_level):
    if risk_level != "High":
        return

    os.makedirs("logs", exist_ok=True)
    alert_file = "logs/alerts.csv"

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "alert_message": "Immediate maintenance required",
        "predicted_RUL": predicted_rul,
        "risk_level": risk_level,
        "cycle": input_data.get("cycle")
    }
    df = pd.DataFrame([row])

    if not os.path.exists(alert_file):
        df.to_csv(alert_file, index=False)
    else:
        df.to_csv(alert_file, mode="a", header=False, index=False)


@app.route("/")
def home():
    return jsonify({
        "message": "Turbofan RUL Prediction API is running",
        "model": MODEL_NAME,
        "model_uri": MODEL_URI,
        "version": MODEL_VERSION,
        "available_endpoints": ["/predict", "/dashboard"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON body. Expected a single JSON object."}), 400

    missing_columns = [col for col in FEATURE_COLUMNS if col not in data]
    if missing_columns:
        return jsonify({"error": "Missing required features", "missing_columns": missing_columns}), 400

    input_df = pd.DataFrame([data])
    input_df = input_df[FEATURE_COLUMNS]

    prediction = model.predict(input_df)[0]
    prediction = max(0, round(float(prediction), 2))
    risk_level = get_risk_level(prediction)

    log_prediction(data, prediction, risk_level)
    log_alert(data, prediction, risk_level)

    response = {
        "predicted_RUL": prediction,
        "risk_level": risk_level,
        "model_version": MODEL_VERSION
    }
    if risk_level == "High":
        response["alert"] = "Immediate maintenance required"

    return jsonify(response)


@app.route("/dashboard")
def dashboard():
    log_file = "logs/prediction_logs.csv"
    alert_file = "logs/alerts.csv"

    if not os.path.exists(log_file):
        return """
        <html><head><title>Turbofan Monitoring Dashboard</title></head>
        <body style="font-family: Arial; padding: 30px;">
            <h1>Turbofan Monitoring Dashboard</h1>
            <p>No prediction logs found yet.</p>
        </body></html>
        """

    df = pd.read_csv(log_file)
    total_predictions = len(df)
    average_rul = round(df["predicted_RUL"].mean(), 2)
    high_risk_count = int((df["risk_level"] == "High").sum())
    medium_risk_count = int((df["risk_level"] == "Medium").sum())
    low_risk_count = int((df["risk_level"] == "Low").sum())

    alert_count = 0
    if os.path.exists(alert_file):
        alert_count = len(pd.read_csv(alert_file))

    latest_columns = [
        col for col in ["timestamp", "model_version", "cycle", "predicted_RUL", "risk_level"]
        if col in df.columns
    ]
    latest_rows = df[latest_columns].tail(10).to_html(index=False)

    html = f"""
    <html>
    <head>
        <title>Turbofan Monitoring Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f6f8; padding: 30px; }}
            h1 {{ color: #1f2937; }}
            .subtitle {{ color: #4b5563; margin-bottom: 25px; }}
            .cards {{ display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }}
            .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); width: 220px; }}
            .card h2 {{ margin: 0; font-size: 28px; }}
            .card p {{ color: #555; }}
            .high {{ border-left: 6px solid #dc2626; }}
            .medium {{ border-left: 6px solid #f59e0b; }}
            .low {{ border-left: 6px solid #16a34a; }}
            table {{ background: white; border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; text-align: center; }}
            th {{ background-color: #1f2937; color: white; }}
        </style>
    </head>
    <body>
        <h1>Turbofan Predictive Maintenance Dashboard</h1>
        <p class="subtitle">Model: {MODEL_NAME} | Version: {MODEL_VERSION}</p>
        <div class="cards">
            <div class="card"><h2>{total_predictions}</h2><p>Total Predictions</p></div>
            <div class="card"><h2>{average_rul}</h2><p>Average Predicted RUL</p></div>
            <div class="card high"><h2>{high_risk_count}</h2><p>High Risk Engines</p></div>
            <div class="card medium"><h2>{medium_risk_count}</h2><p>Medium Risk Engines</p></div>
            <div class="card low"><h2>{low_risk_count}</h2><p>Low Risk Engines</p></div>
            <div class="card high"><h2>{alert_count}</h2><p>Maintenance Alerts</p></div>
        </div>
        <h2>Latest Prediction Logs</h2>
        {latest_rows}
    </body>
    </html>
    """
    return html


if __name__ == "__main__":
    app.run(debug=True, port=8000)
