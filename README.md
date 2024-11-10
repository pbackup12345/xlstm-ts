# 🚀 xLSTM-TS: Extended Long-Short Term Memory for Time Series

![Project Logo](./assets/logo.png)

**Authors:** Gonzalo López Gil, Paul Duhamel-Sebline, Andrew McCarren  
*Published in: [An Evaluation of Deep Learning Models for Stock Market Trend Prediction](https://arxiv.org/abs/2408.12408)*

This repository contains the implementation of the **xLSTM-TS model**, a time series-optimised adaptation of the Extended Long Short-Term Memory (xLSTM) architecture proposed by Beck et al. (2024). The xLSTM-TS model modifies the xLSTM framework to make it suitable for time series forecasting. The architecture includes the xLSTM-TS implementation and integrates wavelet denoising techniques to enhance forecasting accuracy. While designed for versatility in time series forecasting, the model has been applied to short-term Stock Market Trend Prediction, demonstrating its effectiveness in financial applications as a key use case.

In addition to xLSTM-TS, this repository features implementations of several state-of-the-art forecasting models for benchmarking purposes, such as TCN, N-BEATS, TFT, N-HiTS, and TiDE. These models have been evaluated alongside xLSTM-TS in our study. This repository provides datasets and code for the complete workflow, from preprocessing, to model training, and evaluation, along with detailed comparisons of accuracy and trend prediction capabilities.

This is the **official repository** for the paper *"An Evaluation of Deep Learning Models for Stock Market Trend Prediction."*

```bibtex
@misc{gil2024evaluationdeeplearningmodels,
      title={An Evaluation of Deep Learning Models for Stock Market Trend Prediction}, 
      author={Gonzalo Lopez Gil and Paul Duhamel-Sebline and Andrew McCarren},
      year={2024},
      eprint={2408.12408},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.12408}, 
}
```

## 📋 Table of Contents

- [✨ Key Features](#-key-features)
- [📄 Abstract](#-abstract)
- [⚙️ Installation](#-installation)
- [🚀 Usage](#-usage)
- [📊 Dataset](#-dataset)
- [🔧 Preprocessing](#-preprocessing)
- [🧠 Models](#-models)
- [📈 Results](#-results)
- [🤝 Contributions](#-contributions)
- [📚 References](#-references)

## ✨ Key Features

- **xLSTM-TS Implementation**: An adaptation of the Extended LSTM architecture for time series applications.
- **Wavelet Denoising**: Noise reduction using discrete wavelet transforms (DWT) for enhanced signal clarity.
- **Benchmark Models**: Includes leading deep learning architectures for comparison, such as TCN, N-BEATS, TFT, N-HiTS and TiDE.
- **Comprehensive Evaluation**: Includes metrics such as Accuracy, F1 Score, MAE, RMSE, RMSSE, and MASE.
- **Interactive Notebooks**: Experiment with pre-defined setups or customise parameters to explore your own datasets.
- **Extensible Codebase**: Use the modular code in the src/ml folder for your own projects.

## 📄 Abstract

The stock market is a fundamental component of financial systems, reflecting economic health, providing investment opportunities, and influencing global dynamics. Accurate stock market predictions can lead to significant gains and promote better investment decisions. However, predicting stock market trends is challenging due to their non-linear and stochastic nature.

This study investigates the efficacy of advanced deep learning models for short-term trend forecasting using daily and hourly closing prices from the S&P 500 index and the Brazilian ETF EWZ. The models explored include Temporal Convolutional Networks (TCN), Neural Basis Expansion Analysis for Time Series Forecasting (N-BEATS), Temporal Fusion Transformers (TFT), Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS), and Time-series Dense Encoder (TiDE). Furthermore, we introduce the Extended Long Short-Term Memory for Time Series (xLSTM-TS) model, an xLSTM adaptation optimised for time series prediction.

Wavelet denoising techniques were applied to smooth the signal and reduce minor fluctuations, providing cleaner data as input for all approaches. Denoising significantly improved performance in predicting stock price direction. Among the models tested, xLSTM-TS consistently outperformed others. For example, it achieved a test accuracy of 72.82% and an F1 score of 73.16% on the EWZ daily dataset.

By leveraging advanced deep learning models and effective data preprocessing techniques, this research provides valuable insights into the application of machine learning for market movement forecasting, highlighting both the potential and the challenges involved.

## ⚙️ Installation

To set up the environment and run the code, follow the steps below.

### 🛠️ Clone the Repository

First, clone this repository to your local machine or Google Colab environment:

```bash
git clone https://github.com/gonzalopezgil/xlstm-ts.git
```

Navigate to the repository folder:

```bash
cd xlstm-ts
```

### 📦 Install Dependencies

Install the required Python dependencies:

```bash
pip install -r requirements.txt --quiet
```

### 🗂️ Add the `src` Directory to the Python Path

The project uses a modular structure, and the `src` folder contains all the core code. To ensure the code runs smoothly, you need to add the `src` directory to the Python path. This is necessary for importing utilities and model code into your scripts or notebooks.

Run the following Python code to add the `src` directory to the path:

```python
import sys
import os

# Get the current working directory
current_dir = os.getcwd()

# Construct the path to the 'src' directory
src_path = os.path.join(current_dir, 'src')

# Add the 'src' directory to the Python path
if src_path not in sys.path:
    sys.path.append(src_path)

# Now you can import modules from the 'src' directory
from ml.utils.imports import *  # Example import
```

### 🔑 Optional: Retrieve Additional Hourly Data

If you need hourly stock data beyond 2 years (due to API limitations), you can use the **Tiingo API** to fetch it. Follow these steps:

1. Create a Tiingo account at [Tiingo API](https://api.tiingo.com).
2. Access your API Token at [API Token Account Page](https://api.tiingo.com/account/api/token).
3. Set the token in your environment:

```python
import os
os.environ['TIINGO_API_KEY'] = 'your_tiingo_api_key_here'
```

## 🚀 Usage

### 📓 Using the Jupyter Notebooks

**Note**: A GPU runtime is required for training deep learning models. For example, a T4 GPU in Google Colab works perfectly.

The experiments and examples in this project are provided as Jupyter notebooks under the `notebooks` folder.

To run an experiment:

1. Open a notebook from the `notebooks` folder.
2. Modify the **Constants** section at the top of the notebook if needed:

```python
# Dataset settings
TICKER = '^GSPC'  # S&P 500 index
STOCK = 'S&P 500'

# Date range and frequency
START_DATE = '2000-01-01'
END_DATE = '2023-12-31'
FREQ = '1d'  # daily frequency

# Train, validation, test split
TRAIN_END_DATE = '2021-01-01'
VAL_END_DATE = '2022-07-01'
```

3. Run the notebook to train and evaluate models or explore your own datasets.

### 🖥️ Using the Source Code

If you want to integrate xLSTM-TS or other models into your own project:

1.	Explore the `src/ml` folder, which contains all model and workflow implementations, including training, evaluation, and preprocessing scripts.
2.	The `src/gui` folder contains an app under development, not yet complete.

## 📊 Dataset

The datasets used in this study include daily and hourly data for two indices: the S&P 500 and the Brazilian ETF EWZ. These were sourced from the Yahoo Finance and Tiingo APIs, with data spanning the following periods:

- **EWZ Daily**: 14/07/2000 - 29/12/2023
- **S&P 500 Daily**: 03/01/2000 - 29/12/2023
- **EWZ Hourly**: 13/07/2020 - 11/07/2024
- **S&P 500 Hourly**: 13/07/2020 - 11/07/2024

The data fields include Date, High, Low, Close, Adjusted Close, and Volume, with the **Close** price serving as the target variable for trend prediction.

## 🔧 Preprocessing

### 🌀 Noise Reduction

To improve model accuracy, we used **wavelet denoising** to remove noise from the time series data. This approach, based on the discrete wavelet transform (DWT), effectively reduced minor fluctuations, providing a clearer signal for model training.

### 📐 Data Splitting

Each dataset was divided into training, validation, and test sets to enable robust model evaluation:

- **Daily Data**: Training (86%), Validation (7%), Test (7%)
- **Hourly Data**: Training (75%), Validation (12.5%), Test (12.5%)

## 🧠 Models

The focus of this repository is the xLSTM-TS model, an adaptation of the **Extended Long Short-Term Memory (xLSTM)** architecture, as proposed by Beck et al. (2024), specifically designed for time series applications. To benchmark its performance, we also implemented several well-regarded deep learning models for time series forecasting using the Darts library:

1. **xLSTM-TS (Extended LSTM for Time Series)** - Our proposed model that adapts the xLSTM architecture for time series forecasting.
2. **Temporal Convolutional Network (TCN)** - Uses causal convolutions for capturing temporal dependencies.
3. **Neural Basis Expansion Analysis for Time Series Forecasting (N-BEATS)** - Employs residual connections to capture both short- and long-term trends in time series data.
4. **Temporal Fusion Transformer (TFT)** - Integrates attention mechanisms for interpretability in multi-horizon forecasting.
5. **Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS)** - A hierarchical approach optimised for long-horizon forecasts.
6. **Time-series Dense Encoder (TiDE)** - Combines dense encoders with MLPs for efficient predictive modelling.

## 📈 Results

### 📊 Performance Metrics

The models were evaluated using several metrics, including Accuracy, F1 Score, MAE, RMSE, RMSSE, and MASE. The xLSTM-TS model consistently outperformed other models, demonstrating robust predictive power across both daily and hourly datasets. Below are some highlights:

- **EWZ Daily**: xLSTM-TS achieved a Test Accuracy of 72.87% and an F1 Score of 73.16%.
- **S&P 500 Daily**: xLSTM-TS achieved a Test Accuracy of 71.28% and an F1 Score of 73.00%.

These results show that xLSTM-TS is especially effective in capturing stock market trends when paired with wavelet-based denoising.

### 🗝️ Key Findings

- **Wavelet Denoising**: This preprocessing technique greatly enhanced prediction accuracy by providing a clearer input signal.
- **Model Performance**: xLSTM-TS showed superior performance across datasets, especially when compared with other state-of-the-art models.
- **Timeframe Sensitivity**: Predictions of daily trends generally achieved higher accuracy than hourly trends, likely due to the higher volatility in shorter time frames.

## 🤝 Contributions

- **Gonzalo López Gil** - School of Computing, Dublin City University (Dublin, Ireland)  
  Email: gonzalo.lopezgil2@mail.dcu.ie
- **Paul Duhamel-Sebline** - School of Computing, Dublin City University (Dublin, Ireland)  
  Email: paul.duhamelsebline2@mail.dcu.ie
- **Andrew McCarren** - Insight Centre for Data Analytics, Dublin City University (Dublin, Ireland)  
  Email: andrew.mccarren@dcu.ie

## 📚 References

For citing this work, please use the following references:

### 📝 Our Paper

```bibtex
@misc{gil2024evaluationdeeplearningmodels,
      title={An Evaluation of Deep Learning Models for Stock Market Trend Prediction}, 
      author={Gonzalo Lopez Gil and Paul Duhamel-Sebline and Andrew McCarren},
      year={2024},
      eprint={2408.12408},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.12408}, 
}
```

### 📑 Original xLSTM Paper

```bibtex
@misc{beck2024xlstmextendedlongshortterm,
      title={xLSTM: Extended Long Short-Term Memory}, 
      author={Maximilian Beck and Korbinian Pöppel and Markus Spanring and Andreas Auer and Oleksandra Prudnikova and Michael Kopp and Günter Klambauer and Johannes Brandstetter and Sepp Hochreiter},
      year={2024},
      eprint={2405.04517},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.04517}, 
}
```
