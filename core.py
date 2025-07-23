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
    """Handle data loading and preprocessing"""
    
    def __init__(self, sequence_length=60, prediction_horizon=288):  # 288 = 24h * 60min / 5min
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
        self.ti = TechnicalIndicators()
    
    def load_and_preprocess(self, file_path, asset_name):
        """Load CSV and create features"""
        print(f"Loading {asset_name} data from {file_path}")
        
        # Load data - expecting columns: timestamp, open, high, low, close, volume
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure all required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create features (this now handles NaN values properly)
        df = self._create_features(df)
        
        # Reset index after feature creation
        df = df.reset_index(drop=True)
        
        print(f"Loaded {len(df)} records for {asset_name}")
        print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    
    def _create_features(self, df):
        """Create technical indicators and price-based features"""
        # Basic price features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_change'] = df['volume'].pct_change()
        
        # Technical indicators
        df['rsi'] = self.ti.rsi(df['close'])
        upper_bb, middle_bb, lower_bb = self.ti.bollinger_bands(df['close'])
        df['bb_upper'] = upper_bb
        df['bb_middle'] = middle_bb
        df['bb_lower'] = lower_bb
        df['bb_width'] = (upper_bb - lower_bb) / middle_bb
        
        macd_line, signal_line, histogram = self.ti.macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Moving averages
        mas = self.ti.moving_averages(df['close'])
        for ma_name, ma_values in mas.items():
            df[ma_name] = ma_values
            df[f'{ma_name}_ratio'] = df['close'] / ma_values
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Volatility features
        df['volatility'] = df['close'].rolling(window=20).std()
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Handle NaN values more intelligently
        # First, fill NaN values with forward fill, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # If there are still NaN values, fill with 0 for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        return df
    
    def create_sequences(self, df, asset_name):
        """Create sequences for training"""
        feature_columns = [col for col in df.columns if col not in ['timestamp']]
        
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[feature_columns])
        self.scalers[asset_name] = scaler
        
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(scaled_data) - self.prediction_horizon):
            # Input sequence
            seq = scaled_data[i - self.sequence_length:i]
            sequences.append(seq)
            
            # Target: next 288 close prices (24h forecast)
            close_idx = feature_columns.index('close')
            target = scaled_data[i:i + self.prediction_horizon, close_idx]
            targets.append(target)
        
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
    """LSTM model for price prediction"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, prediction_horizon=288):
        super(PricePredictionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, prediction_horizon)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last output for prediction
        out = attn_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class PricePredictor:
    """Main predictor class"""
    
    def __init__(self, sequence_length=60, prediction_horizon=288):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.preprocessor = DataPreprocessor(sequence_length, prediction_horizon)
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create models directory
        self.models_dir = "./models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
        print(f"Models will be saved to: {self.models_dir}")
    
    def prepare_data(self, file_paths):
        """Prepare data for all assets"""
        self.data = {}
        self.sequences = {}
        self.targets = {}
        self.feature_columns = {}
        
        for asset_name, file_path in file_paths.items():
            # Load and preprocess
            df = self.preprocessor.load_and_preprocess(file_path, asset_name)
            self.data[asset_name] = df
            
            # Create sequences
            sequences, targets, feature_cols = self.preprocessor.create_sequences(df, asset_name)
            self.sequences[asset_name] = sequences
            self.targets[asset_name] = targets
            self.feature_columns[asset_name] = feature_cols
            
            print(f"Created {len(sequences)} sequences for {asset_name}")
    
    def create_datasets(self, asset_name, train_split=0.7, val_split=0.2):
        """Create train/val/test datasets"""
        sequences = self.sequences[asset_name]
        targets = self.targets[asset_name]
        
        # Chronological split
        train_size = int(len(sequences) * train_split)
        val_size = int(len(sequences) * val_split)
        
        train_seq = sequences[:train_size]
        train_targets = targets[:train_size]
        
        val_seq = sequences[train_size:train_size + val_size]
        val_targets = targets[train_size:train_size + val_size]
        
        test_seq = sequences[train_size + val_size:]
        test_targets = targets[train_size + val_size:]
        
        train_dataset = PriceDataset(train_seq, train_targets)
        val_dataset = PriceDataset(val_seq, val_targets)
        test_dataset = PriceDataset(test_seq, test_targets)
        
        return train_dataset, val_dataset, test_dataset
    
    def train_model(self, asset_name, epochs=100, batch_size=32, lr=0.001):
        """Train model for specific asset"""
        print(f"\nTraining model for {asset_name}")
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = self.create_datasets(asset_name)
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = len(self.feature_columns[asset_name])
        model = PricePredictionLSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            prediction_horizon=self.prediction_horizon
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_sequences, batch_targets in train_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_sequences, batch_targets in val_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    outputs = model(batch_sequences)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(self.models_dir, f'best_model_{asset_name}.pth')
                torch.save(model.state_dict(), model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model
        model_path = os.path.join(self.models_dir, f'best_model_{asset_name}.pth')
        model.load_state_dict(torch.load(model_path))
        self.models[asset_name] = model
        
        print(f"✅ Model saved to: {model_path}")
        return train_losses, val_losses
    
    def load_model(self, asset_name):
        """Load a pre-trained model for an asset"""
        model_path = os.path.join(self.models_dir, f'best_model_{asset_name}.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Initialize model with correct input size
        if asset_name not in self.feature_columns:
            raise ValueError(f"Feature columns not found for {asset_name}. Please prepare data first.")
        
        input_size = len(self.feature_columns[asset_name])
        model = PricePredictionLSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            prediction_horizon=self.prediction_horizon
        ).to(self.device)
        
        # Load model weights
        model.load_state_dict(torch.load(model_path))
        self.models[asset_name] = model
        
        print(f"✅ Loaded model from: {model_path}")
        return model
    
    def predict(self, asset_name, sequence):
        """Make prediction for given sequence"""
        # Load model if not already loaded
        if asset_name not in self.models:
            self.load_model(asset_name)
        
        model = self.models[asset_name]
        model.eval()
        
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            prediction = model(sequence_tensor)
            prediction = prediction.cpu().numpy().flatten()
        
        return prediction
    
    def predict_next_24h(self, asset_name, current_data):
        """Predict next 24 hours from current data"""
        # Preprocess current data
        df = pd.DataFrame(current_data)
        df = self.preprocessor._create_features(df)
        
        # Get feature columns
        feature_columns = self.feature_columns[asset_name]
        
        # Scale data
        scaler = self.preprocessor.scalers[asset_name]
        scaled_data = scaler.transform(df[feature_columns].dropna())
        
        # Get last sequence
        if len(scaled_data) >= self.sequence_length:
            sequence = scaled_data[-self.sequence_length:]
            
            # Make prediction
            prediction_scaled = self.predict(asset_name, sequence)
            
            # Inverse transform for close price
            close_idx = feature_columns.index('close')
            dummy_array = np.zeros((len(prediction_scaled), len(feature_columns)))
            dummy_array[:, close_idx] = prediction_scaled
            prediction_original = scaler.inverse_transform(dummy_array)[:, close_idx]
            
            return prediction_original
        else:
            raise ValueError(f"Need at least {self.sequence_length} data points")
    
    def evaluate_model(self, asset_name):
        """Evaluate model performance"""
        _, _, test_dataset = self.create_datasets(asset_name)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = self.models[asset_name]
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_sequences, batch_targets in test_loader:
                batch_sequences = batch_sequences.to(self.device)
                outputs = model(batch_sequences)
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
        rmse = np.sqrt(mean_squared_error(all_targets.flatten(), all_predictions.flatten()))
        
        print(f"\n{asset_name} Model Performance:")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        
        return mae, rmse
