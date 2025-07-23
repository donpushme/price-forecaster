from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    @staticmethod
    def rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    @staticmethod
    def bollinger_bands(prices, window=20, num_std=2):
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band
    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    @staticmethod
    def moving_averages(prices, windows=[5, 10, 20, 50]):
        mas = {}
        for window in windows:
            mas[f'MA_{window}'] = prices.rolling(window=window).mean()
        return mas

class DataPreprocessor:
    def __init__(self, sequence_length=60, prediction_horizon=288):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
        self.ti = TechnicalIndicators()
    def load_and_preprocess(self, file_path):
        print(f"Loading bitcoin data from {file_path}")
        df = pd.read_csv(file_path)
        print(f"Loaded raw data: {len(df)} rows")
        # Drop volume column if present
        if 'volume' in df.columns:
            df = df.drop(columns=['volume'])
        # Drop rows with missing OHLC
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        print(f"After dropping rows with missing OHLC: {len(df)} rows")
        if len(df) == 0:
            raise ValueError(f"No valid OHLC data found in {file_path}")
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        print(f"After timestamp cleaning: {len(df)} rows")
        # Ensure all required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        # Validate price data only
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df = df[df[col] > 0]
        print(f"After price validation: {len(df)} rows")
        if len(df) < 100:
            raise ValueError(f"Insufficient data after cleaning: {len(df)} rows (need at least 100)")
        # Create features
        df = self._create_features(df)
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped_rows = initial_len - len(df)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with NaN values after feature creation")
        if len(df) == 0:
            raise ValueError(f"All data was dropped due to NaN values. Check data quality in {file_path}")
        print(f"Final dataset: {len(df)} records for bitcoin")
        print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    def _create_features(self, df):
        price_change_calc = df['close'].pct_change()
        df['price_change'] = price_change_calc.fillna(0) if hasattr(price_change_calc, 'fillna') else 0.0
        high_low_ratio_calc = df['high'] / df['low']
        df['high_low_ratio'] = high_low_ratio_calc.fillna(1.0) if hasattr(high_low_ratio_calc, 'fillna') else 1.0
        price_range_calc = (df['high'] - df['low']) / df['close']
        df['price_range'] = price_range_calc.fillna(0) if hasattr(price_range_calc, 'fillna') else 0.0
        open_close_ratio_calc = df['open'] / df['close']
        df['open_close_ratio'] = open_close_ratio_calc.fillna(1.0) if hasattr(open_close_ratio_calc, 'fillna') else 1.0
        if len(df) >= 14:
            rsi_values = self.ti.rsi(df['close'])
            df['rsi'] = rsi_values.fillna(50.0) if hasattr(rsi_values, 'fillna') else 50.0
        else:
            df['rsi'] = 50.0
        if len(df) >= 20:
            upper_bb, middle_bb, lower_bb = self.ti.bollinger_bands(df['close'])
            df['bb_upper'] = upper_bb.fillna(df['close']) if hasattr(upper_bb, 'fillna') else df['close']
            df['bb_middle'] = middle_bb.fillna(df['close']) if hasattr(middle_bb, 'fillna') else df['close']
            df['bb_lower'] = lower_bb.fillna(df['close']) if hasattr(lower_bb, 'fillna') else df['close']
            if hasattr(middle_bb, 'notna'):
                df['bb_width'] = np.where(
                    middle_bb.notna() & (middle_bb != 0),
                    (upper_bb - lower_bb) / middle_bb,
                    0
                )
            else:
                df['bb_width'] = 0.0
            bb_range = upper_bb - lower_bb
            if hasattr(bb_range, 'fillna'):
                df['bb_position'] = np.where(
                    bb_range != 0,
                    (df['close'] - lower_bb) / bb_range,
                    0.5
                )
            else:
                df['bb_position'] = 0.5
        else:
            df['bb_upper'] = df['close']
            df['bb_middle'] = df['close']
            df['bb_lower'] = df['close']
            df['bb_width'] = 0.0
            df['bb_position'] = 0.5
        if len(df) >= 26:
            macd_line, signal_line, histogram = self.ti.macd(df['close'])
            df['macd'] = macd_line.fillna(0.0) if hasattr(macd_line, 'fillna') else 0.0
            df['macd_signal'] = signal_line.fillna(0.0) if hasattr(signal_line, 'fillna') else 0.0
            df['macd_histogram'] = histogram.fillna(0.0) if hasattr(histogram, 'fillna') else 0.0
        else:
            df['macd'] = 0.0
            df['macd_signal'] = 0.0
            df['macd_histogram'] = 0.0
        mas = self.ti.moving_averages(df['close'], windows=[5, 10, 20, 50])
        for ma_name, ma_values in mas.items():
            window = int(ma_name.split('_')[1])
            if len(df) >= window:
                df[ma_name] = ma_values.fillna(df['close']) if hasattr(ma_values, 'fillna') else df['close']
                if hasattr(ma_values, 'notna'):
                    df[f'{ma_name}_ratio'] = np.where(
                        ma_values.notna() & (ma_values != 0),
                        df['close'] / ma_values,
                        1.0
                    )
                else:
                    df[f'{ma_name}_ratio'] = 1.0
                momentum_calc = df['close'] - ma_values
                df[f'{ma_name}_momentum'] = momentum_calc.fillna(0) if hasattr(momentum_calc, 'fillna') else 0.0
            else:
                df[ma_name] = df['close']
                df[f'{ma_name}_ratio'] = 1.0
                df[f'{ma_name}_momentum'] = 0.0
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(float)
        df['hour_of_day'] = df['timestamp'].dt.hour / 24.0
        if len(df) >= 20:
            volatility_values = df['close'].rolling(window=20).std()
            df['volatility'] = volatility_values.fillna(0.0) if hasattr(volatility_values, 'fillna') else 0.0
            df['volatility_ratio'] = np.where(
                df['close'] != 0,
                df['volatility'] / df['close'],
                0
            )
        else:
            df['volatility'] = 0.0
            df['volatility_ratio'] = 0.0
        if len(df) >= 5:
            momentum_5_calc = df['close'] / df['close'].shift(5) - 1
            df['momentum_5'] = momentum_5_calc.fillna(0) if hasattr(momentum_5_calc, 'fillna') else 0.0
        else:
            df['momentum_5'] = 0.0
        if len(df) >= 10:
            momentum_10_calc = df['close'] / df['close'].shift(10) - 1
            df['momentum_10'] = momentum_10_calc.fillna(0) if hasattr(momentum_10_calc, 'fillna') else 0.0
        else:
            df['momentum_10'] = 0.0
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        df['is_green'] = (df['close'] > df['open']).astype(float)
        return df

class BitcoinPredictionModel(nn.Module):
    """
    Advanced Bitcoin price prediction model using LSTM with attention mechanism
    Predicts OHLC prices and statistical moments for Monte Carlo simulation
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 3, 
                 dropout: float = 0.2, output_steps: int = 288):  # 288 = 24hrs * 12 (5min intervals)
        super(BitcoinPredictionModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        # Feature extraction layers
        self.feature_norm = nn.LayerNorm(input_size)
        
        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Price prediction heads
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 4 * output_steps)  # OHLC for each timestep
        )
        
        # Statistical moments prediction heads
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_steps)  # Standard deviation
        )
        
        self.skewness_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_steps)  # Skewness
        )
        
        self.kurtosis_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_steps)  # Kurtosis
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        
        # Normalize input features
        x = self.feature_norm(x)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last timestep for predictions
        last_hidden = attended_out[:, -1, :]
        
        # Generate predictions
        prices = self.price_head(last_hidden)
        volatility = torch.relu(self.volatility_head(last_hidden)) + 1e-6  # Ensure positive
        skewness = self.skewness_head(last_hidden)
        kurtosis = torch.relu(self.kurtosis_head(last_hidden)) + 3.0  # Ensure kurtosis >= 3
        
        # Reshape prices to (batch_size, output_steps, 4) for OHLC
        prices = prices.view(batch_size, self.output_steps, 4)
        
        return {
            'prices': prices,
            'volatility': volatility,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

class BitcoinTrainer:
    """
    Training class for Bitcoin prediction model
    """
    
    def __init__(self, model: BitcoinPredictionModel, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
    def custom_loss(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Custom loss function combining price prediction and statistical moments"""
        
        # Price prediction loss (MSE)
        price_loss = nn.MSELoss()(predictions['prices'], targets)
        
        # Calculate actual statistical moments from target returns
        target_returns = torch.diff(targets[:, :, 3], dim=1)  # Returns from close prices
        target_std = torch.std(target_returns, dim=1, keepdim=True)
        target_skew = self._calculate_skewness(target_returns)
        target_kurt = self._calculate_kurtosis(target_returns)
        
        # Statistical moments losses
        vol_loss = nn.MSELoss()(predictions['volatility'][:, :-1], target_std.squeeze())
        skew_loss = nn.MSELoss()(predictions['skewness'][:, :-1], target_skew)
        kurt_loss = nn.MSELoss()(predictions['kurtosis'][:, :-1], target_kurt)
        
        # Combined loss
        total_loss = price_loss + 0.1 * vol_loss + 0.05 * skew_loss + 0.05 * kurt_loss
        
        return total_loss, {
            'price_loss': price_loss.item(),
            'vol_loss': vol_loss.item(),
            'skew_loss': skew_loss.item(),
            'kurt_loss': kurt_loss.item()
        }
    
    def _calculate_skewness(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate skewness"""
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        skew = torch.mean(((x - mean) / std) ** 3, dim=1)
        return skew
    
    def _calculate_kurtosis(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate kurtosis"""
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        kurt = torch.mean(((x - mean) / std) ** 4, dim=1)
        return kurt
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_losses = {'total': 0, 'price': 0, 'vol': 0, 'skew': 0, 'kurt': 0}
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(data)
            
            loss, loss_components = self.custom_loss(predictions, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_losses['total'] += loss.item()
            total_losses['price'] += loss_components['price_loss']
            total_losses['vol'] += loss_components['vol_loss']
            total_losses['skew'] += loss_components['skew_loss']
            total_losses['kurt'] += loss_components['kurt_loss']
            num_batches += 1
        
        return {k: v / num_batches for k, v in total_losses.items()}
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_losses = {'total': 0, 'price': 0, 'vol': 0, 'skew': 0, 'kurt': 0}
        num_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                predictions = self.model(data)
                
                loss, loss_components = self.custom_loss(predictions, target)
                
                total_losses['total'] += loss.item()
                total_losses['price'] += loss_components['price_loss']
                total_losses['vol'] += loss_components['vol_loss']
                total_losses['skew'] += loss_components['skew_loss']
                total_losses['kurt'] += loss_components['kurt_loss']
                num_batches += 1
        
        return {k: v / num_batches for k, v in total_losses.items()}

class MonteCarloSimulator:
    """
    Monte Carlo simulation for Bitcoin price forecasting
    """
    
    def __init__(self, model: BitcoinPredictionModel, processor: DataPreprocessor, device: str = 'cpu'):
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        
    def simulate_paths(self, initial_sequence: torch.Tensor, num_simulations: int = 1000) -> Dict[str, np.ndarray]:
        """Generate Monte Carlo price paths"""
        self.model.eval()
        
        with torch.no_grad():
            # Get model predictions
            predictions = self.model(initial_sequence.to(self.device))
            
            # Extract predictions
            mean_prices = predictions['prices'].cpu().numpy()  # (batch, steps, 4)
            volatility = predictions['volatility'].cpu().numpy()  # (batch, steps)
            skewness = predictions['skewness'].cpu().numpy()  # (batch, steps)
            kurtosis = predictions['kurtosis'].cpu().numpy()  # (batch, steps)
            
            batch_size, steps, _ = mean_prices.shape
            
            # Generate Monte Carlo paths
            mc_paths = np.zeros((batch_size, num_simulations, steps, 4))
            
            for b in range(batch_size):
                for sim in range(num_simulations):
                    # Initialize with predicted means
                    path = mean_prices[b].copy()
                    
                    # Add stochastic component based on predicted moments
                    for step in range(steps):
                        # Generate random shocks with predicted statistical properties
                        shock = self._generate_distribution_shock(
                            volatility[b, step], 
                            skewness[b, step], 
                            kurtosis[b, step]
                        )
                        
                        # Apply shock to prices (multiplicative for returns)
                        if step > 0:
                            returns_shock = shock * 0.01  # Scale shock
                            path[step] = path[step-1] * (1 + returns_shock)
                        else:
                            path[step] = path[step] * (1 + shock * 0.001)
                    
                    mc_paths[b, sim] = path
            
            return {
                'paths': mc_paths,
                'mean_prediction': mean_prices,
                'volatility': volatility,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
    
    def _generate_distribution_shock(self, volatility: float, skewness: float, kurtosis: float) -> float:
        """Generate random shock with specified statistical moments"""
        # Use Cornish-Fisher expansion to approximate distribution
        normal_shock = np.random.normal(0, 1)
        
        # Adjust for skewness and kurtosis
        adjusted_shock = (
            normal_shock + 
            (skewness / 6) * (normal_shock**2 - 1) +
            (kurtosis - 3) / 24 * (normal_shock**3 - 3*normal_shock)
        )
        
        return adjusted_shock * volatility

# Example usage and training script
def main():
    """Main training and prediction pipeline"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create sample data (replace with actual Bitcoin OHLC data)
    print("Creating sample data...")
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='5min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(len(dates)).cumsum() + 50000,
        'high': np.random.randn(len(dates)).cumsum() + 50500,
        'low': np.random.randn(len(dates)).cumsum() + 49500,
        'close': np.random.randn(len(dates)).cumsum() + 50000,
        'volume': np.random.exponential(1000, len(dates))
    })
    
    # Initialize data processor
    processor = DataPreprocessor(sequence_length=144)
    
    # Create technical features
    print("Creating technical features...")
    processed_data = processor.load_and_preprocess('sample_data.csv') # Assuming a sample file path
    
    # Prepare sequences
    print("Preparing sequences...")
    X, y = processor.prepare_sequences(processed_data)
    
    print(f"Created {len(X)} sequences with shape {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_size = X.shape[2]
    model = BitcoinPredictionModel(input_size=input_size)
    
    # Initialize trainer
    trainer = BitcoinTrainer(model)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(100):
        train_losses = trainer.train_epoch(train_loader)
        val_losses = trainer.validate(val_loader)
        
        trainer.scheduler.step(val_losses['total'])
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_losses['total']:.6f} | Val Loss: {val_losses['total']:.6f}")
        
        # Early stopping
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print("Early stopping triggered")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Monte Carlo simulation example
    print("\nRunning Monte Carlo simulation...")
    simulator = MonteCarloSimulator(model, processor)
    
    # Use first test sample for simulation
    test_input = torch.FloatTensor(X_test[:1])  # Single batch
    mc_results = simulator.simulate_paths(test_input, num_simulations=500)
    
    print(f"Generated {mc_results['paths'].shape[1]} Monte Carlo paths")
    print(f"Price prediction shape: {mc_results['mean_prediction'].shape}")
    print(f"Volatility shape: {mc_results['volatility'].shape}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot Monte Carlo paths
    plt.subplot(2, 2, 1)
    paths = mc_results['paths'][0, :50, :, 3]  # First 50 simulations, close prices
    for i in range(50):
        plt.plot(paths[i], alpha=0.3, color='blue')
    plt.plot(mc_results['mean_prediction'][0, :, 3], color='red', linewidth=2, label='Mean Prediction')
    plt.title('Monte Carlo Price Paths (Close)')
    plt.legend()
    
    # Plot volatility
    plt.subplot(2, 2, 2)
    plt.plot(mc_results['volatility'][0])
    plt.title('Predicted Volatility')
    
    # Plot skewness
    plt.subplot(2, 2, 3)
    plt.plot(mc_results['skewness'][0])
    plt.title('Predicted Skewness')
    
    # Plot kurtosis
    plt.subplot(2, 2, 4)
    plt.plot(mc_results['kurtosis'][0])
    plt.title('Predicted Kurtosis')
    
    plt.tight_layout()
    plt.savefig('bitcoin_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nModel training and simulation completed!")
    print("Results saved to 'bitcoin_prediction_results.png'")

if __name__ == "__main__":
    main()