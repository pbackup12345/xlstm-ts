# src/models.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from darts import TimeSeries
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.utils.callbacks import TFMProgressBar
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.darts_preprocessing import inverse_normalise_data_darts
from src.constants import SEQ_LENGTH, RANDOM_STATE

# -------------------------------------------------------------------------------------------
# Functions for Darts models
# -------------------------------------------------------------------------------------------

# early stopping (needs to be reset for each model later on)
# this setting stops training once the the validation loss has not decreased by more than 1e-3 for 10 epochs
early_stopping_args = {
    "monitor": "val_loss",
    "patience": 10,
    "min_delta": 1e-3,
    "mode": "min",
}

optimizer_kwargs = {
    "lr": 1e-4,
}

# learning rate scheduler
lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
lr_scheduler_kwargs = {
    "gamma": 0.999,
}

# PyTorch Lightning Trainer arguments
pl_trainer_kwargs = {
    "gradient_clip_val": 1,
    "max_epochs": 200,
    "accelerator": "auto",
    "callbacks": [],
}

common_model_args = {
    "input_chunk_length": SEQ_LENGTH,  # lookback window
    "output_chunk_length": 1,  # forecast/lookahead window
    "optimizer_kwargs": optimizer_kwargs,
    "pl_trainer_kwargs": pl_trainer_kwargs,
    "lr_scheduler_cls": lr_scheduler_cls,
    "lr_scheduler_kwargs": lr_scheduler_kwargs,
    "likelihood": None,  # use a likelihood for probabilistic forecasts
    "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
    "force_reset": True,
    "batch_size": 256,
    "random_state": RANDOM_STATE,
}

# Common parameters for SOME models
def create_params(input_chunk_length: int, output_chunk_length: int, full_training=True, likelihood=QuantileRegression()):
    # early stopping: this setting stops training once the the validation
    # loss has not decreased by more than 1e-5 for 10 epochs
    early_stopper = EarlyStopping(
        **early_stopping_args
    )

    # PyTorch Lightning Trainer arguments
    if full_training:
        limit_train_batches = None
        limit_val_batches = None
        max_epochs = 200
        batch_size = 256
    else:
        limit_train_batches = 20
        limit_val_batches = 10
        max_epochs = 40
        batch_size = 64

    # only show the training and prediction progress bars
    progress_bar = TFMProgressBar(
        enable_sanity_check_bar=False, enable_validation_bar=False
    )
    pl_trainer_kwargs = {
        "gradient_clip_val": 1,
        "max_epochs": max_epochs,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "accelerator": "auto",
        "callbacks": [early_stopper, progress_bar],
    }

    # for probabilistic models, we use quantile regression, and set `loss_fn` to `None`
    if likelihood:
      loss_fn = None
    else:
      loss_fn = torch.nn.MSELoss()


    return {
        "input_chunk_length": input_chunk_length,  # lookback window
        "output_chunk_length": output_chunk_length,  # forecast/lookahead window
        "use_reversible_instance_norm": True,
        "optimizer_kwargs": optimizer_kwargs,
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "likelihood": likelihood,  # use a `likelihood` for probabilistic forecasts
        "loss_fn": loss_fn,  # use a `loss_fn` for determinsitic model
        "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
        "force_reset": True,
        "batch_size": batch_size,
        "random_state": RANDOM_STATE,
        "add_encoders": {
            "cyclic": {
                "future": ["hour", "dayofweek", "month"]
            }  # add cyclic time axis encodings as future covariates
        },
    }

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

def descale_data(predictions, series, scaler):
    backtest = inverse_normalise_data_darts(predictions, scaler)
    actual_values = inverse_normalise_data_darts(series, scaler)

    return backtest, actual_values

def fit(model, train, val):
    model.fit(series=train, val_series=val, verbose=False)

# -------------------------------------------------------------------------------------------
# Forecasting metrics
#
# https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
# -------------------------------------------------------------------------------------------

def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    # Naive forecasting method which just repeats previous samples
    return actual[:-seasonality]

def root_mean_squared_scaled_error(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    # RMSSE - Root Mean Squared Scaled Error
    q = mean_squared_error(actual, predicted) / mean_squared_error(actual[seasonality:], _naive_forecasting(actual, seasonality))
    return np.sqrt(q)

def mean_absolute_scaled_error(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    # MASE - Mean Absolute Scaled Error
    return mean_absolute_error(actual, predicted) / mean_absolute_error(actual[seasonality:], _naive_forecasting(actual, seasonality))

def calculate_metrics(actual, prediction, model_name, data_type):
    # Convert Darts TimeSeries to NumPy arrays if necessary
    if isinstance(actual, TimeSeries):
        actual = actual.values().flatten()
    if isinstance(prediction, TimeSeries):
        prediction = prediction.values().flatten()

    # Calculate metrics using scikit-learn functions
    metrics = {
        "MAE": mean_absolute_error(actual, prediction),
        "MSE": mean_squared_error(actual, prediction),
        "RMSE": root_mean_squared_error(actual, prediction),
        "RMSSE": root_mean_squared_scaled_error(actual, prediction),
        "MAPE": mean_absolute_percentage_error(actual, prediction) * 100,
        "MASE": mean_absolute_scaled_error(actual, prediction),
        "R2": r2_score(actual, prediction)
    }

    for metric_name, metric_value in metrics.items():
        value_str = f"{metric_value:.2f}"
        if metric_name == "MAPE":
            value_str += "%"
        print(f"{model_name} ({data_type}) | {metric_name}: {value_str}")

    return metrics

def visualise(actual, prediction, stock, model_name, data_type, show_complete=True, dates=None):
    if isinstance(actual, TimeSeries):
        dates = actual.time_index
        actual = actual.values().flatten()
    if isinstance(prediction, TimeSeries):
        prediction = prediction.values().flatten()

    title = f"{model_name} Predictions (Trained with "
    if data_type == "Original":
        title += "Original Data)"
    elif data_type == "Denoised":
        title += "Denoised Data)"
    else:
        title = f"{model_name} Predictions"

    if not show_complete:
        sample = 30
        dates = dates[:sample]
        actual = actual[:sample]
        prediction = prediction[:sample]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label='Original Data', color='black', linestyle='-', linewidth=2)
    plt.plot(dates, prediction, label='1-day Forecast', color='grey', linestyle='--', linewidth=2)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=17)
    plt.xlabel('Date', fontsize=17, weight='bold')
    plt.ylabel(f'{stock} Close Price', fontsize=17, weight='bold')
    plt.title(title, fontsize=15)

    plt.grid(False)
    plt.show()

# -------------------------------------------------------------------------------------------
# Predict directions
# -------------------------------------------------------------------------------------------

def calculate_directions(data):
    directions = np.diff(data.squeeze())
    directional_data = np.zeros((directions.size, 2))
    for i, diff in enumerate(directions):
        if diff > 0:
            directional_data[i] = [0, 1]  # Up
        else:
            directional_data[i] = [1, 0]  # Down
    return directional_data

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes, cbar=False)

    # Add percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.55, f'\n({cm_norm[i, j]:.2%})',
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='black',
                     fontsize=9)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion matrix')
    plt.show()

def calculate_movement_metrics(true_labels, predicted_labels, model_name, set_type, data_type):
    if set_type == "Train":
        # Calculate only accuracy for the training set
        accuracy = accuracy_score(true_labels, predicted_labels) * 100
        print(f'{model_name} ({data_type}) | Train Accuracy: {accuracy:.2f}%')

        return {'Train Accuracy': accuracy}

    if set_type == "Val":
        # Calculate only accuracy for the validation set
        accuracy = accuracy_score(true_labels, predicted_labels) * 100
        print(f'{model_name} ({data_type}) | Validation Accuracy: {accuracy:.2f}%')

        return {'Validation Accuracy': accuracy}

    elif set_type == "Test":
        # Calculate metrics for the test set
        accuracy = accuracy_score(true_labels, predicted_labels) * 100
        recall = recall_score(true_labels, predicted_labels, pos_label=1) * 100
        precision_rise = precision_score(true_labels, predicted_labels, pos_label=1) * 100
        precision_fall = precision_score(true_labels, predicted_labels, pos_label=0) * 100
        f1 = f1_score(true_labels, predicted_labels, pos_label=1) * 100

        print(f'{model_name} ({data_type}) | Test Accuracy: {accuracy:.2f}%')
        print(f'{model_name} ({data_type}) | Recall: {recall:.2f}%')
        print(f'{model_name} ({data_type}) | Precision (Rise): {precision_rise:.2f}%')
        print(f'{model_name} ({data_type}) | Precision (Fall): {precision_fall:.2f}%')
        print(f'{model_name} ({data_type}) | F1 Score: {f1:.2f}%')

        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=["Down", "Up"], yticklabels=["Down", "Up"], cbar=False)

        # Add percentages
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.55, f'\n({cm_norm[i, j]:.2%})',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='black',
                         fontsize=9)

        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{model_name} Confusion Matrix ({data_type} Data)')
        plt.show()

        return {
            'Test Accuracy': accuracy,
            'Recall': recall,
            'Precision (Rise)': precision_rise,
            'Precision (Fall)': precision_fall,
            'F1 Score': f1
        }

def evaluate_directional_movement(actual_values_train, backtest_train, actual_values_val, backtest_val, actual_values_test, backtest_test, model_name, data_type, using_darts=True):
    if using_darts:
        train_y = actual_values_train.values()
        train_predictions = backtest_train.values()
        val_y = actual_values_val.values()
        val_predictions = backtest_val.values()
        test_y = actual_values_test.values()
        test_predictions = backtest_test.values()
    else:
        train_y = actual_values_train
        train_predictions = backtest_train
        val_y = actual_values_val
        val_predictions = backtest_val
        test_y = actual_values_test
        test_predictions = backtest_test

    # Calculate directions for training set
    true_directions_train = calculate_directions(train_y)
    predicted_directions_train = calculate_directions(train_predictions)

    # Convert to class labels for training set
    true_labels_train = np.argmax(true_directions_train, axis=1)
    predicted_labels_train = np.argmax(predicted_directions_train, axis=1)

    # Calculate directions for validation set
    true_directions_val = calculate_directions(val_y)
    predicted_directions_val = calculate_directions(val_predictions)

    # Convert to class labels for validation set
    true_labels_val = np.argmax(true_directions_val, axis=1)
    predicted_labels_val = np.argmax(predicted_directions_val, axis=1)

    # Calculate directions for test set
    true_directions_test = calculate_directions(test_y)
    predicted_directions_test = calculate_directions(test_predictions)

    # Convert to class labels for test set
    true_labels_test = np.argmax(true_directions_test, axis=1)
    predicted_labels_test = np.argmax(predicted_directions_test, axis=1)

    # Calculate metrics for training set
    metrics_train = calculate_movement_metrics(true_labels_train, predicted_labels_train, model_name, "Train", data_type)

    # Calculate metrics for validation set
    metrics_val = calculate_movement_metrics(true_labels_val, predicted_labels_val, model_name, "Val", data_type)

    # Calculate metrics for test set
    metrics_test = calculate_movement_metrics(true_labels_test, predicted_labels_test, model_name, "Test", data_type)

    # Combine accuracy into the test metrics dictionary
    metrics_test['Validation Accuracy'] = metrics_val['Validation Accuracy']
    metrics_test['Train Accuracy'] = metrics_train['Train Accuracy']

    return true_labels_test, predicted_labels_test, metrics_test

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
