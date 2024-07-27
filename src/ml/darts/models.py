# src/ml/darts/models.py

from darts.models import TCNModel, NBEATSModel, TFTModel, TiDEModel, NHiTSModel, TSMixerModel
from src.ml.constants import SEQ_LENGTH, RANDOM_STATE, FULL_TRAINING
from src.ml.models import create_params, common_model_args

# -------------------------------------------------------------------------------------------
# TCN
#
# References:
#
# - Paper (2018): https://doi.org/10.48550/arXiv.1803.01271
# - Code: https://github.com/unit8co/darts/blob/master/darts/models/forecasting/tcn_model.py
# - Darts documentation: https://unit8co.github.io/darts/examples/05-TCN-examples.html
# -------------------------------------------------------------------------------------------

def get_model_tcn():
    model_tcn = TCNModel(
        model_name="TCN",

        # Common parameters
        n_epochs = 20,
        input_chunk_length = SEQ_LENGTH,
        output_chunk_length = 1,
        random_state = RANDOM_STATE,

        # Specific parameters
        dropout = 0,
        dilation_base = 2,
        weight_norm = True,
        kernel_size = 7,
        num_filters = 4
    )

    return model_tcn

# -------------------------------------------------------------------------------------------
# DeepTCN
#
# References:
#
# - Paper (2020): https://doi.org/10.48550/arXiv.1906.04397
# - Code (same as TCN): https://github.com/unit8co/darts/blob/master/darts/models/forecasting/tcn_model.py
# - Darts documentation: https://unit8co.github.io/darts/examples/09-DeepTCN-examples.html
# -------------------------------------------------------------------------------------------

def get_model_deeptcn():
    model_deeptcn = TCNModel(
        model_name="DeepTCN",

        # Common parameters
        batch_size=32,
        n_epochs=20,
        optimizer_kwargs={"lr": 1e-3},
        random_state=RANDOM_STATE,
        input_chunk_length=SEQ_LENGTH,
        output_chunk_length=1,

        # Specific parameters
        dropout=0.2,
        kernel_size=3,
        num_filters=4,

        save_checkpoints=True,
        force_reset=True,
    )

    return model_deeptcn

# -------------------------------------------------------------------------------------------
# N-BEATS
#
# References:
# - Paper (2020): https://doi.org/10.48550/arXiv.1905.10437
# - Code: https://github.com/unit8co/darts/blob/master/darts/models/forecasting/nbeats.py
# - Darts documentation: https://unit8co.github.io/darts/examples/07-NBEATS-examples.html
# -------------------------------------------------------------------------------------------

def get_model_nbeats():
    model_nbeats = NBEATSModel(
        model_name="N-BEATS",

        # Specific parameters
        generic_architecture=True,
        num_stacks=10,
        num_blocks=1,
        num_layers=4,
        layer_widths=512,
        nr_epochs_val_period=1,

        **create_params(
            SEQ_LENGTH,
            1,
            full_training=FULL_TRAINING,
        )
    )

    return model_nbeats

# -------------------------------------------------------------------------------------------
# TFT
# (Temporal Fusion Transformer Model)
# 
# References:
# 
# - Paper (2021): https://doi.org/10.1016/j.ijforecast.2021.03.012
# - Code: https://github.com/unit8co/darts/blob/master/darts/models/forecasting/tft_model.py
# - Darts documentation: https://unit8co.github.io/darts/examples/13-TFT-examples.html
# -------------------------------------------------------------------------------------------

def get_model_tft():
    model_tft = TFTModel(
        model_name="TFT",

        # Specific parameters
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        add_relative_index=True,

        **create_params(
            SEQ_LENGTH,
            1,
            full_training=FULL_TRAINING,
        )
    )

    return model_tft

# -------------------------------------------------------------------------------------------
# TiDE
# (TimeSeries Dense Encoder Model)
# 
# References:
# - Paper (2024): https://doi.org/10.48550/arXiv.2304.08424
# - Code: https://github.com/unit8co/darts/blob/master/darts/models/forecasting/tide_model.py
# - Darts documentation: https://unit8co.github.io/darts/examples/18-TiDE-examples.html
# 
# RIN
# (Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift)
# 
# - Paper: https://openreview.net/forum?id=cGDAkQo1C0p
# -------------------------------------------------------------------------------------------

def get_model_tide():
    model_tide = TiDEModel(
        model_name="TiDE",
        use_reversible_instance_norm=True,
        **common_model_args,
    )

    return model_tide

# -------------------------------------------------------------------------------------------
# N-HiTS 
# (Neural Hierarchical Interpolation for Time Series Forecasting)
# 
# References:
# 
# - Paper (2022): https://doi.org/10.48550/arXiv.2201.12886
# - Code: https://github.com/unit8co/darts/blob/master/darts/models/forecasting/nhits.py
# - Darts documentation: https://unit8co.github.io/darts/examples/18-TiDE-examples.html
# -------------------------------------------------------------------------------------------

def get_model_nhits():
    model_nhits = NHiTSModel(
        model_name="N-HiTS",

        **create_params(
            SEQ_LENGTH,
            1,
            full_training=FULL_TRAINING,
        )
    )

    return model_nhits

# -------------------------------------------------------------------------------------------
# TSMixer
# (TimeSeries Mixer Model)
# 
# References:
# - Paper (2023): https://doi.org/10.48550/arXiv.2303.06053
# - Code: https://github.com/ditschuk/pytorch-tsmixer
# - Darts documentation: https://unit8co.github.io/darts/examples/21-TSMixer-examples.html
# -------------------------------------------------------------------------------------------

def get_model_tsm():
    model_tsm = TSMixerModel(
        model_name="TSMixer",

        **create_params(
        SEQ_LENGTH,
        1,
        full_training=FULL_TRAINING,
        ),
    )

    return model_tsm
