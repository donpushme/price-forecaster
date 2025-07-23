# PyTorch Price Forecasting

A machine learning system for cryptocurrency price forecasting using LSTM with attention mechanism.

## Features

- Fetches historical price data from Pyth Network API
- Processes OHLCV (Open, High, Low, Close, Volume) data
- Calculates technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- Uses LSTM with attention mechanism for price prediction
- Predicts next 24 hours of price movements (288 5-minute intervals)
- Saves trained models in organized directory structure
- **Real-time forecasting system** with live price updates every minute

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Check and fetch training data:
```bash
python check_and_fetch_data.py
```

3. Train the models:
```bash
python train_models.py
```

4. Start real-time forecasting:
```bash
python real_time_forecast.py
```

## Data Format

The system expects CSV files with the following columns:
- `timestamp`: DateTime in format 'YYYY-MM-DD HH:MM:SS'
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

## API Response Format

The system handles Pyth Network API responses in the format:
```json
{
  "s": "ok",
  "t": [1735707300, 1735707600],
  "o": [93447.18141577, 93357.85438592],
  "h": [93471.38962908, 93390.0],
  "l": [93325.2474621, 93345.19861641],
  "c": [93357.87633587, 93383.75891115],
  "v": [0.0, 0.0]
}
```

## Model Architecture

- **Input**: 60 time steps of OHLCV data + technical indicators
- **LSTM Layers**: 2 layers with 128 hidden units
- **Attention**: Multi-head attention mechanism
- **Output**: 288 predictions (24 hours of 5-minute intervals)
- **Loss**: Mean Squared Error (MSE)

## Technical Indicators

The model includes the following technical indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (5, 10, 20, 50 periods)
- Price change ratios
- Volatility measures
- Time-based features

## Directory Structure

```
pytorch-price-forecast/
├── training_data/
│   ├── bitcoin_5min.csv
│   ├── ethereum_5min.csv
│   └── xau_5min.csv
├── models/
│   ├── best_model_bitcoin.pth
│   ├── best_model_ethereum.pth
│   └── best_model_xau.pth
├── forecasts/
│   ├── bitcoin_forecast_20240115_1430.csv
│   ├── ethereum_forecast_20240115_1430.csv
│   └── xau_forecast_20240115_1430.csv
├── fetch_training_data.py
├── model_train.py
├── real_time_forecast.py
├── test_data_fetch.py
├── test_real_time.py
└── requirements.txt
```

## Usage

### Training
```python
from model_train import PricePredictor

# Initialize predictor
predictor = PricePredictor(sequence_length=60, prediction_horizon=288)

# Prepare data
file_paths = {
    'bitcoin': './training_data/bitcoin_5min.csv',
    'ethereum': './training_data/ethereum_5min.csv',
    'xau': './training_data/xau_5min.csv'
}
predictor.prepare_data(file_paths)

# Train model
train_losses, val_losses = predictor.train_model('bitcoin', epochs=100)
```

### Loading Pre-trained Models
```python
# Load a pre-trained model
predictor.load_model('bitcoin')
```

### Prediction
```python
# Make prediction using last 60 data points
last_60_data = predictor.data['bitcoin'].tail(60).to_dict('records')
prediction = predictor.predict_next_24h('bitcoin', last_60_data)
```

### Real-Time Forecasting
```python
from real_time_forecast import RealTimeForecaster

# Initialize real-time forecaster
forecaster = RealTimeForecaster()

# Run continuous forecasting (updates every minute)
forecaster.run_forecast()
```

## Real-Time Forecasting System

The real-time forecasting system provides:

### **Features**:
- **Live Price Updates**: Fetches current prices every minute
- **24-Hour Predictions**: Generates 289 price points (current + 288 future)
- **5-Minute Intervals**: Predictions at 5-minute intervals
- **Multiple Assets**: Bitcoin, Ethereum, and Gold (XAU)
- **Automatic Saving**: Saves forecasts to CSV files
- **Continuous Operation**: Runs indefinitely until stopped

### **Output**:
- **Console Display**: Shows current prices and predictions
- **CSV Files**: Saves complete forecasts with timestamps
- **Real-time Updates**: Updates every 60 seconds

### **Sample Output**:
```
🕐 Current time: 2024-01-15 14:30:00 UTC
Fetching current prices...
💰 bitcoin: $42,150.50
💰 ethereum: $2,450.75
💰 xau: $2,025.30

📊 Generating 24-hour forecasts...

🔮 Forecasting for BITCOIN
Current Price: $42,150.50
Current Time: 2024-01-15 14:30:00 UTC

First 10 predictions:
  14:35:00: $42,165.75
  14:40:00: $42,180.25
  14:45:00: $42,195.40
  ...

💾 Forecast saved to: ./forecasts/bitcoin_forecast_20240115_1430.csv
⏰ Next update in 60 seconds...
```

## Model Saving

- **Location**: All trained models are saved in the `./models/` directory
- **Naming**: Models are named as `best_model_{asset_name}.pth`
- **Automatic**: Models are automatically saved during training when validation loss improves
- **Loading**: Models can be loaded using `predictor.load_model(asset_name)`

## Files

- `fetch_training_data.py`: Fetches historical data from Pyth Network
- `model_train.py`: Main training script with LSTM model
- `real_time_forecast.py`: Real-time forecasting system
- `test_data_fetch.py`: Tests data format and quality
- `test_real_time.py`: Tests real-time forecasting functionality
- `requirements.txt`: Python dependencies
- `training_data/`: Directory containing fetched CSV data
- `models/`: Directory containing trained model files
- `forecasts/`: Directory containing real-time forecast outputs

## Notes

- The model uses 5-minute intervals for training
- Predictions are made for the next 24 hours (288 intervals)
- Early stopping is implemented to prevent overfitting
- The system automatically handles data scaling and feature engineering
- Models are saved in PyTorch format (.pth) for easy loading and deployment
- Real-time system requires trained models and historical data to function
- Forecasts are saved with timestamps for easy tracking and analysis 