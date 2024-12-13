# src/ml/models/xlstm_ts/logic.py

import pandas as pd
from ml.models.shared.visualisation import visualise
from ml.models.shared.metrics import calculate_metrics
from ml.models.shared.directional_prediction import evaluate_directional_movement
from ml.models.xlstm_ts.preprocessing import inverse_normalise_data_xlstm
from ml.constants import SEQ_LENGTH_XLSTM
from ml.models.xlstm_ts.xlstm_ts_model import create_xlstm_model
from ml.models.xlstm_ts.training import train_model, evaluate_model

# -------------------------------------------------------------------------------------------
# xLSTM-TS logic
# -------------------------------------------------------------------------------------------

def run_xlstm_ts(train_x, train_y, val_x, val_y, test_x, test_y, scaler, stock, data_type, test_dates, train_y_original=None, val_y_original=None, test_y_original=None):
    xlstm_stack, input_projection, output_projection = create_xlstm_model(SEQ_LENGTH_XLSTM)

    xlstm_stack, input_projection, output_projection = train_model(xlstm_stack, input_projection, output_projection, train_x, train_y, val_x, val_y)

    test_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, test_x)

    # Invert the normalisation for comparison
    test_predictions = inverse_normalise_data_xlstm(test_predictions.squeeze(), scaler)

    f = open('./res.csv',"w")

    for idx in range(1,len(test_predictions)): 
      if idx == 0:
        continue
      # Access test_dates using .iloc to get the value at the integer position idx
      print(pd.to_datetime(test_dates.values[idx]).strftime('%Y-%m-%d') + " " + ("up" if test_predictions[idx][0] - test_predictions[idx-1][0] > 0 else "down")+" "+str(test_predictions[idx][0]))
      f.write(pd.to_datetime(test_dates.values[idx]).strftime('%Y-%m-%d') + "\t" + ("up" if test_predictions[idx][0] - test_predictions[idx-1][0] > 0 else "down")+"\t"+str(test_predictions[idx][0] - test_predictions[idx-1][0])+"\n")
    
    f.close()

    test_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, test_x)
    test_predictions = inverse_normalise_data_xlstm(test_predictions.squeeze(), scaler)

    f = open('./res.csv',"w")

    for idx in range(1,len(test_predictions)): 
      if idx == 0:
        continue
      # Access test_dates using .iloc to get the value at the integer position idx
      print(pd.to_datetime(test_dates.values[idx]).strftime('%Y-%m-%d') + " " + ("up" if test_predictions[idx][0] - test_predictions[idx-1][0] > 0 else "down")+" "+str(test_predictions[idx][0]))
      f.write(pd.to_datetime(test_dates.values[idx]).strftime('%Y-%m-%d') + "\t" + ("up" if test_predictions[idx][0] - test_predictions[idx-1][0] > 0 else "down")+"\t"+str(test_predictions[idx][0] - test_predictions[idx-1][0])+"\n")
    
    f.close()

    # If the original data is provided, use it for the evaluation
    if train_y_original is not None and val_y_original is not None and test_y_original is not None:
        train_y = train_y_original
        val_y = val_y_original
        test_y = test_y_original
    
    test_y = inverse_normalise_data_xlstm(test_y, scaler)

    model_name = 'xLSTM-TS'
    metrics_price = calculate_metrics(test_y, test_predictions, model_name, data_type)

    visualise(test_y, test_predictions, stock, model_name, data_type, show_complete=True, dates=test_dates)
    visualise(test_y, test_predictions, stock, model_name, data_type, show_complete=False, dates=test_dates)

    train_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, train_x)
    val_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, val_x)

    # Invert the normalisation for comparison
    train_predictions = inverse_normalise_data_xlstm(train_predictions.squeeze(), scaler)
    train_y = inverse_normalise_data_xlstm(train_y, scaler)

    val_predictions = inverse_normalise_data_xlstm(val_predictions.squeeze(), scaler)
    val_y = inverse_normalise_data_xlstm(val_y, scaler)

    true_labels, predicted_labels, metrics_direction = evaluate_directional_movement(train_y, train_predictions, val_y, val_predictions, test_y, test_predictions, model_name, data_type, using_darts=False)

    metrics_price.update(metrics_direction)

    # Combine data into a DataFrame
    data = {
        'Date': test_dates.tolist()[:-1],
        'Close': [item for sublist in test_y for item in sublist][:-1],
        'Predicted Value': [item for sublist in test_predictions for item in sublist][:-1],
        'True Label': true_labels.tolist(),
        'Predicted Label': predicted_labels.tolist()
    }

    results_df = pd.DataFrame(data)

    return results_df, metrics_price
