# src/imports.py

import datetime
import yfinance as yf

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging
import warnings
import os

import pywt

from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from darts import TimeSeries
from darts.dataprocessing.transformers.scaler import Scaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from darts.models import TCNModel, NBEATSModel, TFTModel, TiDEModel, NHiTSModel, TSMixerModel
from darts.utils.callbacks import TFMProgressBar
from darts.utils.likelihood_models import QuantileRegression

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

# Suppress the specific UserWarning about torch.nn.utils.weight_norm
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.nn.utils.weight_norm is deprecated.*")

logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)

sns.set(style='whitegrid')
