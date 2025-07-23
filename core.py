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
import os
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Calculate technical indicators for price data"""
    
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
    """Handle data loading and preprocessing - FIXED VERSION"""
    
    def __init__(self, sequence_length=60, prediction_horizon=288):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
        self.ti = TechnicalIndicators()
    
    def load_and_preprocess(self, file_path, asset_name):
        """Load CSV and create features - BEST PRACTICE: drop NaNs after feature creation, no filling for training, IGNORE VOLUME"""
        print(f"Loading {asset_name} data from {file_path}")
        df = pd.read_csv(file_path)
        print(f"Loaded raw data: {len(df)} rows")
        
        # Basic data validation and cleaning (ignore volume)
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        print(f"After dropping rows with missing OHLC: {len(df)} rows")
        if len(df) == 0:
            raise ValueError(f"No valid OHLC data found in {file_path}")
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        print(f"After timestamp cleaning: {len(df)} rows")
        
        # Ensure all required columns exist (volume not required)
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate price data only
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df = df[df[col] > 0]  # Remove zero or negative prices
        print(f"After price validation: {len(df)} rows")
        if len(df) < 100:
            raise ValueError(f"Insufficient data after cleaning: {len(df)} rows (need at least 100)")
        
        # Drop volume column if present
        if 'volume' in df.columns:
            df = df.drop(columns=['volume'])
        
        # Create features
        df = self._create_features(df)
        initial_len = len(df)
        # BEST PRACTICE: Drop all rows with any NaN after feature creation
        df = df.dropna().reset_index(drop=True)
        dropped_rows = initial_len - len(df)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with NaN values after feature creation")
        if len(df) == 0:
            raise ValueError(f"All data was dropped due to NaN values. Check data quality in {file_path}")
        print(f"Final dataset: {len(df)} records for {asset_name}")
        print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    
    def _create_features(self, df):
        """Create technical indicators and price-based features - IGNORE VOLUME"""
        # Ensure we have enough data for technical indicators
        if len(df) < 50:  # Minimum for longest MA window
            print("Warning: Dataset too small for all technical indicators")
        # Basic price features (NO VOLUME FEATURES)
        price_change_calc = df['close'].pct_change()
        df['price_change'] = price_change_calc.fillna(0) if hasattr(price_change_calc, 'fillna') else 0.0
        high_low_ratio_calc = df['high'] / df['low']
        df['high_low_ratio'] = high_low_ratio_calc.fillna(1.0) if hasattr(high_low_ratio_calc, 'fillna') else 1.0
        price_range_calc = (df['high'] - df['low']) / df['close']
        df['price_range'] = price_range_calc.fillna(0) if hasattr(price_range_calc, 'fillna') else 0.0
        open_close_ratio_calc = df['open'] / df['close']
        df['open_close_ratio'] = open_close_ratio_calc.fillna(1.0) if hasattr(open_close_ratio_calc, 'fillna') else 1.0
        # Technical indicators with conditional calculation
        if len(df) >= 14:  # Minimum for RSI
            rsi_values = self.ti.rsi(df['close'])
            df['rsi'] = rsi_values.fillna(50.0) if hasattr(rsi_values, 'fillna') else 50.0
        else:
            df['rsi'] = 50.0  # Neutral value
        if len(df) >= 20:  # Minimum for Bollinger Bands
            upper_bb, middle_bb, lower_bb = self.ti.bollinger_bands(df['close'])
            df['bb_upper'] = upper_bb.fillna(df['close']) if hasattr(upper_bb, 'fillna') else df['close']
            df['bb_middle'] = middle_bb.fillna(df['close']) if hasattr(middle_bb, 'fillna') else df['close']
            df['bb_lower'] = lower_bb.fillna(df['close']) if hasattr(lower_bb, 'fillna') else df['close']
            # Safe calculation for bb_width
            if hasattr(middle_bb, 'notna'):
                df['bb_width'] = np.where(
                    middle_bb.notna() & (middle_bb != 0), 
                    (upper_bb - lower_bb) / middle_bb, 
                    0
                )
            else:
                df['bb_width'] = 0.0
            # Safe calculation for bb_position
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
        if len(df) >= 26:  # Minimum for MACD
            macd_line, signal_line, histogram = self.ti.macd(df['close'])
            df['macd'] = macd_line.fillna(0.0) if hasattr(macd_line, 'fillna') else 0.0
            df['macd_signal'] = signal_line.fillna(0.0) if hasattr(signal_line, 'fillna') else 0.0
            df['macd_histogram'] = histogram.fillna(0.0) if hasattr(histogram, 'fillna') else 0.0
        else:
            df['macd'] = 0.0
            df['macd_signal'] = 0.0
            df['macd_histogram'] = 0.0
        # Moving averages with conditional windows
        mas = self.ti.moving_averages(df['close'], windows=[5, 10, 20, 50])
        for ma_name, ma_values in mas.items():
            window = int(ma_name.split('_')[1])
            if len(df) >= window:
                df[ma_name] = ma_values.fillna(df['close']) if hasattr(ma_values, 'fillna') else df['close']
                # Safe ratio calculation
                if hasattr(ma_values, 'notna'):
                    df[f'{ma_name}_ratio'] = np.where(
                        ma_values.notna() & (ma_values != 0), 
                        df['close'] / ma_values, 
                        1.0
                    )
                else:
                    df[f'{ma_name}_ratio'] = 1.0
                # Safe momentum calculation
                momentum_calc = df['close'] - ma_values
                df[f'{ma_name}_momentum'] = momentum_calc.fillna(0) if hasattr(momentum_calc, 'fillna') else 0.0
            else:
                df[ma_name] = df['close']
                df[f'{ma_name}_ratio'] = 1.0
                df[f'{ma_name}_momentum'] = 0.0
        # Time-based features (always calculable)
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)
        # Additional time features
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(float)
        df['hour_of_day'] = df['timestamp'].dt.hour / 24.0
        # Volatility features
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
        # Price momentum features
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
        # Candle pattern features
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        df['is_green'] = (df['close'] > df['open']).astype(float)
        return df
    
    def create_sequences(self, df, asset_name):
        """Create sequences for training - FIXED SCALING"""
        feature_columns = [col for col in df.columns if col not in ['timestamp']]
        
        # CRITICAL FIX: Use MinMaxScaler instead of StandardScaler for price data
        # MinMaxScaler preserves the relative relationships better for financial data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df[feature_columns])
        self.scalers[asset_name] = scaler
        
        # Store original feature order for later use
        self.feature_columns_order = {asset_name: feature_columns}
        
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(scaled_data) - self.prediction_horizon):
            # Input sequence
            seq = scaled_data[i - self.sequence_length:i]
            sequences.append(seq)
            
            # Target: next prediction_horizon close prices
            close_idx = feature_columns.index('close')
            target = scaled_data[i:i + self.prediction_horizon, close_idx]
            targets.append(target)
        
        print(f"Created {len(sequences)} sequences for {asset_name}")
        print(f"Sequence shape: {np.array(sequences).shape}")
        print(f"Target shape: {np.array(targets).shape}")
        
        return np.array(sequences), np.array(targets), feature_columns

class PriceDataset(Dataset):
    """PyTorch Dataset for price sequences"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class PricePredictionLSTM(nn.Module):
    """IMPROVED LSTM model for price prediction"""
    
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3, prediction_horizon=288):
        super(PricePredictionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # Input projection layer
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # LSTM layers with residual connections
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers with skip connections
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size // 2)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, prediction_horizon)
        
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # Project input features
        x = self.input_projection(x)
        
        # LSTM layers with residual connections
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1 + x)  # Residual connection
        lstm_out3, _ = self.lstm3(lstm_out2 + lstm_out1)  # Residual connection
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out3, lstm_out3, lstm_out3)
        
        # Combine LSTM and attention outputs
        combined = lstm_out3 + attn_out  # Residual connection
        
        # Use last output for prediction
        out = combined[:, -1, :]
        out = self.norm1(out)
        
        # Fully connected layers
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.gelu(out)
        out = self.norm2(out)
        
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.gelu(out)
        
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

class PricePredictor:
    """Main predictor class - FIXED VERSION"""
    
    def __init__(self, sequence_length=120, prediction_horizon=288):  # Increased sequence length
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.preprocessor = DataPreprocessor(sequence_length, prediction_horizon)
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create models directory
        self.models_dir = "./models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
        print(f"Sequence length: {sequence_length} (10 hours of 5-min data)")
        print(f"Prediction horizon: {prediction_horizon} (24 hours)")
    
    def prepare_data(self, file_paths):
        """Prepare data for all assets"""
        self.data = {}
        self.sequences = {}
        self.targets = {}
        self.feature_columns = {}
        
        for asset_name, file_path in file_paths.items():
            try:
                # Load and preprocess
                df = self.preprocessor.load_and_preprocess(file_path, asset_name)
                self.data[asset_name] = df
                
                # Create sequences
                sequences, targets, feature_cols = self.preprocessor.create_sequences(df, asset_name)
                
                if len(sequences) == 0:
                    print(f"WARNING: No sequences created for {asset_name}. Check your data.")
                    continue
                
                self.sequences[asset_name] = sequences
                self.targets[asset_name] = targets
                self.feature_columns[asset_name] = feature_cols
                
                print(f"✅ Successfully prepared {asset_name}: {len(sequences)} sequences")
                
            except Exception as e:
                print(f"❌ Error preparing {asset_name}: {str(e)}")
                continue
    
    def create_datasets(self, asset_name, train_split=0.7, val_split=0.2):
        """Create train/val/test datasets with proper chronological split"""
        sequences = self.sequences[asset_name]
        targets = self.targets[asset_name]
        
        # Chronological split (no shuffling to maintain time order)
        train_size = int(len(sequences) * train_split)
        val_size = int(len(sequences) * val_split)
        
        train_seq = sequences[:train_size]
        train_targets = targets[:train_size]
        
        val_seq = sequences[train_size:train_size + val_size]
        val_targets = targets[train_size:train_size + val_size]
        
        test_seq = sequences[train_size + val_size:]
        test_targets = targets[train_size + val_size:]
        
        print(f"Dataset split for {asset_name}:")
        print(f"  Train: {len(train_seq)} sequences")
        print(f"  Val: {len(val_seq)} sequences")
        print(f"  Test: {len(test_seq)} sequences")
        
        train_dataset = PriceDataset(train_seq, train_targets)
        val_dataset = PriceDataset(val_seq, val_targets)
        test_dataset = PriceDataset(test_seq, test_targets)
        
        return train_dataset, val_dataset, test_dataset
    
    def train_model(self, asset_name, epochs=200, batch_size=64, lr=0.0005):
        """Train model with improved parameters"""
        print(f"\n🚀 Training model for {asset_name}")
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = self.create_datasets(asset_name)
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=0, pin_memory=True if self.device.type == 'cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True if self.device.type == 'cuda' else False)
        
        # Initialize improved model
        input_size = len(self.feature_columns[asset_name])
        model = PricePredictionLSTM(
            input_size=input_size,
            hidden_size=256,
            num_layers=3,
            dropout=0.3,
            prediction_horizon=self.prediction_horizon
        ).to(self.device)
        
        # Improved loss and optimizer
        criterion = nn.SmoothL1Loss()  # More robust than MSE for price prediction
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr*10, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (batch_sequences, batch_targets) in enumerate(train_loader):
                batch_sequences = batch_sequences.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_sequences, batch_targets in val_loader:
                    batch_sequences = batch_sequences.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                    outputs = model(batch_sequences)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(self.models_dir, f'best_model_{asset_name}.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': self.preprocessor.scalers[asset_name],
                    'feature_columns': self.feature_columns[asset_name],
                    'config': {
                        'input_size': input_size,
                        'hidden_size': 256,
                        'num_layers': 3,
                        'dropout': 0.3,
                        'prediction_horizon': self.prediction_horizon,
                        'sequence_length': self.sequence_length
                    }
                }, model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 30:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.models[asset_name] = model
        
        print(f"✅ Model training completed and saved to: {model_path}")
        return train_losses, val_losses
    
    def predict_next_24h(self, asset_name, current_data):
        """Prediction: only fill NaNs in the last row, drop NaNs in the rest (best practice)"""
        try:
            # Load model if not in memory
            if asset_name not in self.models:
                self.load_model(asset_name)
            # Convert to DataFrame and preprocess
            if isinstance(current_data, list):
                df = pd.DataFrame(current_data)
            else:
                df = current_data.copy()
            # Ensure timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Create features
            df = self.preprocessor._create_features(df)
            df = df.reset_index(drop=True)
            # Fill NaNs in the last row only (neutral values)
            last_idx = df.index[-1]
            for col in df.columns:
                if pd.isna(df.at[last_idx, col]):
                    if 'rsi' in col:
                        df.at[last_idx, col] = 50.0
                    elif 'macd' in col:
                        df.at[last_idx, col] = 0.0
                    elif 'ma' in col or 'bb_' in col:
                        df.at[last_idx, col] = df.at[last_idx, 'close']
                    else:
                        df.at[last_idx, col] = 0.0
            # Drop NaNs in the rest
            if len(df) > 1:
                df = pd.concat([df.iloc[:-1].dropna(), df.iloc[[-1]]], ignore_index=True)
            else:
                df = df.dropna().reset_index(drop=True)
            if len(df) < self.sequence_length:
                raise ValueError(f"Need at least {self.sequence_length} clean data points, got {len(df)}")
            # Get feature columns and scale data
            feature_columns = self.feature_columns[asset_name]
            scaler = self.preprocessor.scalers[asset_name]
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            scaled_data = scaler.transform(df[feature_columns])
            sequence = scaled_data[-self.sequence_length:]
            # Make prediction
            model = self.models[asset_name]
            model.eval()
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                prediction_scaled = model(sequence_tensor)
                prediction_scaled = prediction_scaled.cpu().numpy().flatten()
            # Inverse transform
            close_idx = feature_columns.index('close')
            dummy_array = np.zeros((len(prediction_scaled), len(feature_columns)))
            dummy_array[:, close_idx] = prediction_scaled
            prediction_original = scaler.inverse_transform(dummy_array)[:, close_idx]
            # Generate timestamps for predictions (5-minute intervals)
            last_timestamp = df['timestamp'].iloc[-1]
            prediction_timestamps = [last_timestamp + pd.Timedelta(minutes=5*(i+1)) for i in range(len(prediction_original))]
            prediction_df = pd.DataFrame({
                'timestamp': prediction_timestamps,
                'predicted_price': prediction_original
            })
            current_price = df['close'].iloc[-1]
            print(f"\n📊 {asset_name} Prediction Summary:")
            print(f"Current price: ${current_price:,.2f}")
            print(f"1-hour prediction: ${prediction_original[11]:,.2f}")
            print(f"6-hour prediction: ${prediction_original[71]:,.2f}")
            print(f"24-hour prediction: ${prediction_original[-1]:,.2f}")
            print(f"Predicted change: {((prediction_original[-1] - current_price) / current_price * 100):+.2f}%")
            return prediction_df
        except Exception as e:
            print(f"❌ Prediction error for {asset_name}: {str(e)}")
            raise
    
    def load_model(self, asset_name):
        """Load pre-trained model"""
        model_path = os.path.join(self.models_dir, f'best_model_{asset_name}.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Restore configuration
        config = checkpoint['config']
        if 'sequence_length' in config:
            del config['sequence_length']
        model = PricePredictionLSTM(**config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore preprocessor state
        self.preprocessor.scalers[asset_name] = checkpoint['scaler']
        self.feature_columns[asset_name] = checkpoint['feature_columns']
        
        self.models[asset_name] = model
        print(f"✅ Loaded model for {asset_name}")
        
        return model

# USAGE EXAMPLE WITH BETTER ERROR HANDLING
def main():
    """Example usage with error handling"""
    
    # File paths
    file_paths = {
        'bitcoin': 'bitcoin_5min.csv',
        'ethereum': 'ethereum_5min.csv', 
        'xau': 'xau_5min.csv'
    }
    
    # Initialize predictor with longer sequence for better context
    predictor = PricePredictor(sequence_length=120, prediction_horizon=288)
    
    try:
        # Prepare data
        print("📥 Preparing data...")
        predictor.prepare_data(file_paths)
        
        # Train models
        for asset_name in predictor.sequences.keys():
            print(f"\n🔥 Training {asset_name}...")
            train_losses, val_losses = predictor.train_model(
                asset_name=asset_name,
                epochs=200,
                batch_size=64,
                lr=0.0005
            )
            
            # Quick evaluation
            print(f"📈 Evaluating {asset_name}...")
            predictor.evaluate_model(asset_name)
        
        print("\n🎉 Training completed for all assets!")
        
    except Exception as e:
        print(f"❌ Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()