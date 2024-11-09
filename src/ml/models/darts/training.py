# src/ml/models/darts/training.py

from ml.models.darts.preprocessing import inverse_normalise_data_darts
from ml.models.shared.metrics import calculate_metrics
from ml.models.shared.directional_prediction import evaluate_directional_movement
from ml.models.shared.visualisation import visualise
from ml.constants import SEQ_LENGTH

# -------------------------------------------------------------------------------------------
# Training and Evaluation functions
# -------------------------------------------------------------------------------------------

def predict(model, series):
    # predict
    backtest = model.historical_forecasts(
        series=series,
        forecast_horizon=1,
        retrain=False,
        verbose=False,
        last_points_only=True
    )

    return backtest

def fit(model, train, val):
    model.fit(series=train, val_series=val, verbose=False)

def descale_data(predictions, series, scaler):
    backtest = inverse_normalise_data_darts(predictions, scaler)
    actual_values = inverse_normalise_data_darts(series, scaler)

    return backtest, actual_values

# -------------------------------------------------------------------------------------------
# Training logic for darts models
# -------------------------------------------------------------------------------------------

def training_darts(model, train, val, test, scaler, stock, data_type, train_denoised=None, val_denoised=None, test_denoised=None, scaler_denoised=None):
    if train_denoised is not None and val_denoised is not None and test_denoised is not None and scaler_denoised is not None:
        # Use denoised data for training and validation if provided
        fit(model, train_denoised, val_denoised)

        backtest_train = predict(model, train_denoised)
        backtest_val = predict(model, val_denoised)
        backtest_test = predict(model, test_denoised)

        backtest_train = inverse_normalise_data_darts(backtest_train, scaler_denoised)
        backtest_val = inverse_normalise_data_darts(backtest_val, scaler_denoised)
        backtest_test = inverse_normalise_data_darts(backtest_test, scaler_denoised)

    else:
        # Use original data for training and validation if denoised data is not provided
        fit(model, train, val)

        backtest_train = predict(model, train)
        backtest_val = predict(model, val)
        backtest_test = predict(model, test)

        backtest_train = inverse_normalise_data_darts(backtest_train, scaler)
        backtest_val = inverse_normalise_data_darts(backtest_val, scaler)
        backtest_test = inverse_normalise_data_darts(backtest_test, scaler)

    model_name = model.model_name

    actual_values_train = train[SEQ_LENGTH:]
    actual_values_train = inverse_normalise_data_darts(actual_values_train, scaler)

    actual_values_val = val[SEQ_LENGTH:]
    actual_values_val = inverse_normalise_data_darts(actual_values_val, scaler)

    actual_values_test = test[SEQ_LENGTH:]
    actual_values_test = inverse_normalise_data_darts(actual_values_test, scaler)

    print("Price Prediction Metrics:\n")

    metrics_price = calculate_metrics(actual_values_test, backtest_test, model_name, data_type)

    visualise(actual_values_test, backtest_test, stock, model_name, data_type, show_complete=True)
    visualise(actual_values_test, backtest_test, stock, model_name, data_type, show_complete=False)

    print("\nDirectional Movement Metrics:\n")

    _, _, metrics_direction = evaluate_directional_movement(actual_values_train, backtest_train, actual_values_val, backtest_val, actual_values_test, backtest_test, model_name, data_type)

    metrics_price.update(metrics_direction)

    return metrics_price
