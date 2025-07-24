import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
from scipy import stats
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class OHLCDataset(Dataset):
    """Dataset for OHLC data with skewness and kurtosis targets"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class SkewnessKurtosisPredictor(nn.Module):
    """Neural Network to predict skewness and kurtosis from OHLC data"""
    
    def __init__(self, input_size=20, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(SkewnessKurtosisPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer for 2 targets (skewness, kurtosis)
        layers.append(nn.Linear(prev_size, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.network(x)

class FinancialFeatureExtractor:
    """Extract financial features from OHLC data"""
    
    @staticmethod
    def calculate_returns(prices):
        """Calculate log returns"""
        return np.log(prices[1:] / prices[:-1])
    
    @staticmethod
    def calculate_volatility(returns, window=20):
        """Calculate rolling volatility"""
        return pd.Series(returns).rolling(window=window, min_periods=1).std().values
    
    @staticmethod
    def calculate_rsi(prices, window=14):
        """Calculate RSI indicator"""
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate([[50], rsi.values])  # Prepend neutral RSI for first value
    
    @staticmethod
    def extract_features_from_ohlc(ohlc_data, lookback_window=5):
        """
        Extract features from OHLC data
        
        Args:
            ohlc_data: List of dictionaries with OHLC data
            lookback_window: Number of periods to look back for features
        
        Returns:
            features: numpy array of features
            targets: numpy array of [skewness, kurtosis] for each period
        """
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlc_data)
        
        # Extract OHLC values
        opens = df['o'].values
        highs = df['h'].values
        lows = df['l'].values
        closes = df['c'].values
        
        features_list = []
        targets_list = []
        
        for i in range(lookback_window, len(df)):
            # Get recent price data for feature calculation
            recent_closes = closes[i-lookback_window:i+1]
            recent_opens = opens[i-lookback_window:i+1]
            recent_highs = highs[i-lookback_window:i+1]
            recent_lows = lows[i-lookback_window:i+1]
            
            # Calculate returns for target calculation
            returns = FinancialFeatureExtractor.calculate_returns(recent_closes)
            
            # Skip if not enough data for statistics
            if len(returns) < 3:
                continue
            
            # Calculate targets (skewness and kurtosis)
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns, fisher=True) + 3  # Convert to excess kurtosis + 3
            
            # Handle infinite or NaN values
            if not (np.isfinite(skewness) and np.isfinite(kurtosis)):
                skewness = 0.0
                kurtosis = 3.0
            
            # Clip extreme values
            skewness = np.clip(skewness, -5, 5)
            kurtosis = np.clip(kurtosis, 1, 10)
            
            # Calculate features
            features = []
            
            # 1. Recent returns (5 features)
            if len(returns) >= 5:
                features.extend(returns[-5:])
            else:
                # Pad with zeros if not enough returns
                padded_returns = np.zeros(5)
                padded_returns[-len(returns):] = returns
                features.extend(padded_returns)
            
            # 2. Price ratios (4 features)
            features.extend([
                recent_highs[-1] / recent_closes[-1] - 1,  # High/Close ratio
                recent_lows[-1] / recent_closes[-1] - 1,   # Low/Close ratio
                recent_closes[-1] / recent_opens[-1] - 1,  # Close/Open ratio
                (recent_highs[-1] - recent_lows[-1]) / recent_closes[-1]  # Range/Close ratio
            ])
            
            # 3. Moving averages (3 features)
            ma_5 = np.mean(recent_closes[-5:]) if len(recent_closes) >= 5 else recent_closes[-1]
            ma_3 = np.mean(recent_closes[-3:]) if len(recent_closes) >= 3 else recent_closes[-1]
            features.extend([
                recent_closes[-1] / ma_5 - 1,  # Price vs MA5
                recent_closes[-1] / ma_3 - 1,  # Price vs MA3
                ma_3 / ma_5 - 1                # MA3 vs MA5
            ])
            
            # 4. Volatility measures (3 features)
            vol_short = np.std(returns[-3:]) if len(returns) >= 3 else np.std(returns)
            vol_long = np.std(returns)
            features.extend([
                vol_short,
                vol_long,
                vol_short / (vol_long + 1e-8) - 1
            ])
            
            # 5. Momentum indicators (3 features)
            momentum_3 = (recent_closes[-1] / recent_closes[-3] - 1) if len(recent_closes) >= 3 else 0
            momentum_5 = (recent_closes[-1] / recent_closes[0] - 1) if len(recent_closes) >= 5 else 0
            features.extend([
                momentum_3,
                momentum_5,
                momentum_3 - momentum_5
            ])
            
            # 6. RSI (1 feature)
            rsi = FinancialFeatureExtractor.calculate_rsi(recent_closes, window=min(14, len(recent_closes)))
            features.append((rsi[-1] - 50) / 50)  # Normalize RSI to [-1, 1]
            
            # 7. Volume proxy (1 feature) - using price range as proxy
            avg_range = np.mean((recent_highs - recent_lows) / recent_closes)
            features.append(avg_range)
            
            features_list.append(features)
            targets_list.append([skewness, kurtosis])
        
        return np.array(features_list), np.array(targets_list)

def parse_data_string(data_string):
    """Parse the JSON string format data"""
    try:
        data = json.loads(data_string)
        return {
            'o': data['o'][0],
            'h': data['h'][0], 
            'l': data['l'][0],
            'c': data['c'][0],
            't': data['t'][0]
        }
    except:
        return None

def create_sample_data(n_samples=1000):
    """Create sample OHLC data for demonstration"""
    np.random.seed(42)
    
    # Start with base price
    base_price = 3400
    data = []
    
    for i in range(n_samples):
        # Generate realistic OHLC data with some randomness
        returns = np.random.normal(0, 0.02)  # 2% daily volatility
        
        open_price = base_price * (1 + returns)
        
        # Generate high, low, close with realistic relationships
        high_mult = 1 + abs(np.random.normal(0, 0.005))
        low_mult = 1 - abs(np.random.normal(0, 0.005))
        close_returns = np.random.normal(0, 0.01)
        
        high_price = open_price * high_mult
        low_price = open_price * low_mult
        close_price = open_price * (1 + close_returns)
        
        # Ensure logical price relationships
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        data.append({
            'o': open_price,
            'h': high_price,
            'l': low_price,
            'c': close_price,
            't': 1753274220 + i * 3600  # Hourly timestamps
        })
        
        base_price = close_price  # Next open follows previous close
    
    return data

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def main():
    # Load real data from bitcoin_5min.csv (exclude volume)
    print("Loading data from bitcoin_5min.csv...")
    df = pd.read_csv('./training_data/bitcoin_5min.csv')
    # Ensure required columns exist
    required_cols = ['timestamp', 'open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # Parse and map to expected format (o/h/l/c/t)
    ohlc_data = []
    for i, row in df.iterrows():
        ohlc_data.append({
            'o': row['open'],
            'h': row['high'],
            'l': row['low'],
            'c': row['close'],
            't': pd.to_datetime(row['timestamp']).timestamp() if not isinstance(row['timestamp'], (int, float)) else row['timestamp']
        })
    print(f"Loaded {len(ohlc_data)} rows from CSV.")
    
    # Extract features
    print("Extracting features...")
    features, targets = FinancialFeatureExtractor.extract_features_from_ohlc(ohlc_data)
    
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Feature statistics - Mean: {np.mean(features, axis=0)[:5]}")
    print(f"Target statistics - Skewness mean: {np.mean(targets[:, 0]):.3f}, Kurtosis mean: {np.mean(targets[:, 1]):.3f}")
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, targets, test_size=0.2, random_state=42
    )
    
    # Create datasets and loaders
    train_dataset = OHLCDataset(X_train, y_train)
    test_dataset = OHLCDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create and train model
    model = SkewnessKurtosisPredictor(input_size=features.shape[1])
    print(f"\nModel architecture:")
    print(model)
    
    print(f"\nTraining model...")
    train_losses, val_losses = train_model(model, train_loader, test_loader)
    
    # Evaluate model
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        predictions = []
        actuals = []
        
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    skew_mse = np.mean((predictions[:, 0] - actuals[:, 0]) ** 2)
    kurt_mse = np.mean((predictions[:, 1] - actuals[:, 1]) ** 2)
    
    print(f"\nTest Results:")
    print(f"Skewness MSE: {skew_mse:.6f}")
    print(f"Kurtosis MSE: {kurt_mse:.6f}")
    print(f"Skewness MAE: {np.mean(np.abs(predictions[:, 0] - actuals[:, 0])):.6f}")
    print(f"Kurtosis MAE: {np.mean(np.abs(predictions[:, 1] - actuals[:, 1])):.6f}")
    
    # Example prediction on new data
    print(f"\nExample predictions:")
    for i in range(5):
        pred_skew, pred_kurt = predictions[i]
        actual_skew, actual_kurt = actuals[i]
        print(f"Sample {i+1}:")
        print(f"  Predicted: Skewness={pred_skew:.3f}, Kurtosis={pred_kurt:.3f}")
        print(f"  Actual:    Skewness={actual_skew:.3f}, Kurtosis={actual_kurt:.3f}")
    
    return model, scaler

# Example usage
if __name__ == "__main__":
    # Example of parsing your data format
    sample_data_string = '{"s":"ok","t":[1753274220],"o":[3423.99],"h":[3424.789],"l":[3423.97],"c":[3424.38]}'
    parsed_data = parse_data_string(sample_data_string)
    print("Parsed data example:", parsed_data)
    
    # Train the model
    model, scaler = main()
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'model_config': {
            'input_size': 20,
            'hidden_sizes': [128, 64, 32],
            'dropout_rate': 0.2
        }
    }, 'skewness_kurtosis_model.pth')
    
    print("\nModel saved as 'skewness_kurtosis_model.pth'")

# Function to load and use the trained model
def load_and_predict(model_path, new_ohlc_data):
    """Load trained model and make predictions on new data"""
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['model_config']
    
    model = SkewnessKurtosisPredictor(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    
    # Extract features from new data
    features, _ = FinancialFeatureExtractor.extract_features_from_ohlc(new_ohlc_data)
    features_scaled = scaler.transform(features)
    
    # Make predictions
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features_scaled)
        predictions = model(features_tensor)
        
    return predictions.numpy()  # Returns [skewness, kurtosis] for each sample