# src/ml/models/xlstm_ts/xlstm_ts_model.py

# -------------------------------------------------------------------------------------------
# New proposed model: xLSTM-TS, a time series-specific implementation
# 
# References:
# 
# - Paper (2024): https://doi.org/10.48550/arXiv.2405.04517
# - Official code: https://github.com/NX-AI/xlstm
# - Parameters for time series: https://github.com/smvorwerk/xlstm-cuda
# -------------------------------------------------------------------------------------------

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
from ml.constants import SEQ_LENGTH_XLSTM
from torchinfo import summary

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
