# src/ml/xlstm_ts/model.py

# -------------------------------------------------------------------------------------------
# New proposed model: xLSTM-TS, a time series-specific implementation
# 
# References:
# 
# - Paper (2024): https://doi.org/10.48550/arXiv.2405.04517
# - Official code: https://github.com/NX-AI/xlstm
# - Parameters for time series: https://github.com/smvorwerk/xlstm-cuda
# -------------------------------------------------------------------------------------------

import pandas as pd
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from src.ml.constants import SEQ_LENGTH_XLSTM
from torchinfo import summary
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.ml.models import visualise, calculate_metrics, evaluate_directional_movement
from src.ml.xlstm_ts.preprocessing import inverse_normalise_data_xlstm

# -------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------

def create_xlstm_model(seq_length):
    # Define your input size, hidden size, and other relevant parameters
    input_size = 1  # Number of features in your time series
    embedding_dim = 64  # Dimension of the embeddings, reduced to save memory
    output_size = 1  # Number of output features (predicting the next value)

    # Define the xLSTM configuration
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, qkv_proj_blocksize=2, num_heads=2  # Reduced parameters to save memory
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=2,  # Reduced number of heads to save memory
                conv1d_kernel_size=2,  # Reduced kernel size to save memory
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.1, act_fn="gelu"),  # Reduced projection factor to save memory
        ),
        context_length=seq_length,
        num_blocks=4,  # Reduced number of blocks to save memory
        embedding_dim=embedding_dim,
        slstm_at=[1],
    )

    # Instantiate the xLSTM stack
    xlstm_stack = xLSTMBlockStack(cfg).to("cuda")

    # Add a linear layer to project input data to the required embedding dimension
    input_projection = nn.Linear(input_size, embedding_dim).to("cuda")

    # Add a final linear layer to project the xLSTM output to the desired output size
    output_projection = nn.Linear(embedding_dim, output_size).to("cuda")

    return xlstm_stack, input_projection, output_projection

# -------------------------------------------------------------------------------------------
# Plot architecture
# -------------------------------------------------------------------------------------------

# Define a simplified model to pass through torchinfo
class ModelWrapper(nn.Module):
    def __init__(self, input_projection, xlstm_stack, output_projection):
        super(ModelWrapper, self).__init__()
        self.input_projection = input_projection
        self.xlstm_stack = xlstm_stack
        self.output_projection = output_projection

    def forward(self, x):
        x = self.input_projection(x)
        x = self.xlstm_stack(x)
        x = self.output_projection(x[:, -1, :])
        return x
    
def plot_architecture_xlstm():
    xlstm_stack, input_projection, output_projection = create_xlstm_model(SEQ_LENGTH_XLSTM)

    model = ModelWrapper(input_projection, xlstm_stack, output_projection).cuda()

    batch_size = 16
    real_input_dimensions = (batch_size, SEQ_LENGTH_XLSTM, 1)

    # Generate the summary with actual input dimensions
    summary(model, input_size=real_input_dimensions)

# -------------------------------------------------------------------------------------------
# Data loader
# -------------------------------------------------------------------------------------------

# DataLoader for batching
def create_dataloader(x, y, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# -------------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------------

def train_model(xlstm_stack, input_projection, output_projection, train_x, train_y, val_x, val_y):
    # Hyperparameters
    learning_rate = 0.0001
    num_epochs = 200
    batch_size = 16  # Reduced batch size to save memory

    best_val_loss = float('inf')
    patience = 40
    trigger_times = 0

    # DataLoader for batching
    train_loader = create_dataloader(train_x, train_y, batch_size, shuffle=True)
    val_loader = create_dataloader(val_x, val_y, batch_size, shuffle=False)

    # Define the loss function and optimiser
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimiser = optim.Adam(list(xlstm_stack.parameters()) + list(input_projection.parameters()) + list(output_projection.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=10)  # Learning rate scheduler

    initial_lr = optimiser.param_groups[0]['lr']
    lr_reduced = False # Flag to track if learning rate has been reduced

    # Training loop
    for epoch in range(num_epochs):
        xlstm_stack.train()  # Set the model to training mode

        for batch_x, batch_y in train_loader:
            # Forward pass
            projected_input_data = input_projection(batch_x)
            xlstm_output = xlstm_stack(projected_input_data)
            predictions = output_projection(xlstm_output[:, -1, :])  # Use the last time step's output

            # Ensure the shapes match
            predictions = predictions.squeeze()
            batch_y = batch_y.squeeze()

            # Compute the loss
            loss = criterion(predictions, batch_y)

            # Backward pass and optimization
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(xlstm_stack.parameters(), max_norm=1.0)  # Apply gradient clipping to prevent the exploding gradient problem
            optimiser.step()

        # Validation step
        xlstm_stack.eval()  # Set the model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                projected_input_data = input_projection(batch_x)
                xlstm_output = xlstm_stack(projected_input_data)
                predictions = output_projection(xlstm_output[:, -1, :])  # Use the last time step's output

                predictions = predictions.squeeze()
                batch_y = batch_y.squeeze()

                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)  # Adjust learning rate based on validation loss

        # Print learning rate reduction only once
        if not lr_reduced and optimiser.param_groups[0]['lr'] < initial_lr:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Reducing learning rate to {optimiser.param_groups[0]["lr"]}')
            lr_reduced = True

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}, Validation Loss: {val_loss:.8f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(xlstm_stack.state_dict(), 'best_model.pth')
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    print("Training complete!")

    return xlstm_stack, input_projection, output_projection

# -------------------------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------------------------

def evaluate_model(xlstm_stack, input_projection, output_projection, test_x):
    # Load the best model
    xlstm_stack.load_state_dict(torch.load('best_model.pth'))

    # Evaluate the model on the test set
    xlstm_stack.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        projected_input_data = input_projection(test_x)
        xlstm_output = xlstm_stack(projected_input_data)
        test_predictions = output_projection(xlstm_output[:, -1, :])  # Use the last time step's output

    return test_predictions

# -------------------------------------------------------------------------------------------
# xLSTM-TS logic
# -------------------------------------------------------------------------------------------

def run_xlstm_ts(train_x, train_y, val_x, val_y, test_x, test_y, scaler, stock, data_type, test_dates, train_y_original=None, val_y_original=None, test_y_original=None):
    xlstm_stack, input_projection, output_projection = create_xlstm_model(SEQ_LENGTH_XLSTM)

    xlstm_stack, input_projection, output_projection = train_model(xlstm_stack, input_projection, output_projection, train_x, train_y, val_x, val_y)

    test_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, test_x)

    # Invert the normalisation for comparison
    test_predictions = inverse_normalise_data_xlstm(test_predictions.squeeze(), scaler)

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
