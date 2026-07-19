
## Project Overview

This project implements an end-to-end machine learning lifecycle management system for predictive maintenance using MLflow. The system predicts the Remaining Useful Life (RUL) of turbofan engines using sensor readings from the NASA C-MAPSS dataset.

The project demonstrates the complete ML lifecycle, including data preprocessing, model training, experiment tracking, hyperparameter tuning, model registry, deployment, monitoring, alert generation, dashboard visualization, and feature importance analysis.

## Domain

Industrial AI / Predictive Maintenance

## Dataset

The project uses the NASA C-MAPSS Turbofan Engine Degradation dataset.

Two dataset subsets were used:

- FD001: one operating condition and one fault mode
- FD003: one operating condition and two fault modes

FD001 was used as the main baseline dataset, while FD003 was used to evaluate the system under a more complex degradation scenario.

## Problem Type

Regression problem.

The target variable is Remaining Useful Life (RUL), which represents the estimated number of operating cycles remaining before engine failure.

## Preprocessing

The preprocessing step reads the training, testing, and true RUL files. For the training data, RUL is calculated using:

```text
RUL = max_cycle - current_cycle
```

To improve model performance, RUL clipping was applied with an upper limit of 125 cycles. This helps the model focus more on the degradation phase closer to engine failure instead of early-life cycles where degradation patterns are less clear.

## Machine Learning Models

The following models were trained and evaluated:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

## Evaluation Metrics

The models were evaluated using:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R2 Score

Lower MAE and RMSE values indicate better performance. A higher R2 score indicates that the model explains more variance in the target variable.

## MLflow Usage

MLflow was used for:

- Experiment tracking
- Parameter logging
- Metric logging
- Model artifact logging
- Hyperparameter tuning comparison
- Model Registry
- Production model aliasing
- Monitoring deployed predictions

## Final Model Results

### FD001 Results

The best model for FD001 was Gradient Boosting.

- MAE: 11.68
- RMSE: 16.17
- R2 Score: 0.8371

### FD003 Results

The best model for FD003 was XGBoost.

- MAE: 11.81
- RMSE: 16.28
- R2 Score: 0.8272

After applying RUL clipping and adding XGBoost, the overall model performance improved significantly compared to the first baseline version.

## Production Model

The best production model was registered in MLflow Model Registry as:

```text
Turbofan_RUL_Predictor
```

Production version:

```text
Version 4
```

Production alias:

```text
production
```

The deployed Flask API loads the model using the production alias:

```text
models:/Turbofan_RUL_Predictor@production
```

## Deployment

A Flask API was created to deploy the registered production model.

The API provides the following endpoints:

### Home Endpoint

```text
GET /
```

Returns a message confirming that the API is running and shows the active model information.

### Prediction Endpoint

```text
POST /predict
```

The endpoint receives engine sensor readings and returns:

- predicted_RUL
- risk_level
- alert message if the engine is high risk

Risk levels are assigned as follows:

- High: RUL <= 30
- Medium: RUL <= 80
- Low: RUL > 80

Example output:

```json
{
  "predicted_RUL": 7.89,
  "risk_level": "High",
  "alert": "Immediate maintenance required"
}
```

## Monitoring Dashboard

A Flask dashboard was added at:

```text
/dashboard
```

The dashboard displays:

- Total predictions
- Average predicted RUL
- High-risk engine count
- Medium-risk engine count
- Low-risk engine count
- Maintenance alert count
- Latest prediction logs

This makes the system closer to a real operational predictive maintenance platform.

## Alert System

The system automatically generates a maintenance alert when the predicted RUL is less than or equal to 30 cycles.

Alerts are stored in:

```text
logs/alerts.csv
```

Prediction logs are stored in:

```text
logs/prediction_logs.csv
```

## Monitoring Script

The monitoring script reads the prediction logs and calculates:

- Total predictions
- Average predicted RUL
- High-risk count
- Medium-risk count
- Low-risk count
- Maintenance alert count

These monitoring metrics are logged back into MLflow under:

```text
Turbofan_RUL_Monitoring
```

## Feature Importance

Feature importance was extracted from the production tree-based model to explain which input features had the largest influence on RUL prediction.

Top important features included:

- sensor_11
- sensor_4
- cycle
- sensor_12
- sensor_7
- sensor_9
- sensor_15

The feature importance results are saved in:

```text
reports/feature_importance.csv
reports/feature_importance.png
```

## Project Structure

```text
PRJ-SulaimanAlareer-2106738/
│
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   ├── train_FD003.txt
│   ├── test_FD003.txt
│   └── RUL_FD003.txt
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── tune.py
│   ├── register_model.py
│   ├── app.py
│   ├── monitor.py
│   └── feature_importance.py
│
├── logs/
│   ├── prediction_logs.csv
│   └── alerts.csv
│
├── reports/
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   └── project_report.docx
│
├── screenshots/
├── mlruns/
├── requirements.txt
└── README.md
```

## How to Run the Project

### 1. Install Requirements

```bash
py -m pip install -r requirements.txt
```

### 2. Test Preprocessing

```bash
py src/preprocess.py
```

### 3. Train Models

```bash
py src/train.py
```

### 4. Run Hyperparameter Tuning

```bash
py src/tune.py
```

### 5. Register the Best Model

```bash
py src/register_model.py
```

### 6. Start MLflow UI

```bash
py -m mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

### 7. Run the Flask API

```bash
py src/app.py
```

Open:

```text
http://127.0.0.1:8000
```

### 8. Open Dashboard

```text
http://127.0.0.1:8000/dashboard
```

### 9. Run Monitoring

```bash
py src/monitor.py
```

### 10. Generate Feature Importance

```bash
py src/feature_importance.py
```

## Conclusion

This project demonstrates a complete MLOps workflow for predictive maintenance. It includes model development, experiment tracking, hyperparameter tuning, model registry, production aliasing, Flask deployment, monitoring, dashboard visualization, alert generation, and feature importance analysis.

The final system is not only a machine learning model, but a small operational predictive maintenance platform that can receive engine sensor data, predict Remaining Useful Life, classify risk level, generate alerts, and monitor prediction activity over time.
