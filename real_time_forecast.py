import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta, timezone
import os
from core import PricePredictor, TechnicalIndicators
import warnings
warnings.filterwarnings('ignore')

class RealTimeForecaster:
    """Real-time price forecasting system"""
    
    def __init__(self):
        self.predictor = PricePredictor(sequence_length=60, prediction_horizon=288)
        self.ti = TechnicalIndicators()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Trading pairs and their file mappings
        self.trading_pairs = {
            'bitcoin': 'Crypto.BTC/USD',
            'ethereum': 'Crypto.ETH/USD',
            'xau': 'Metal.XAU/USD'
        }
        
        # API endpoint for current prices
        self.current_price_url = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
        
        # Historical data files
        self.data_files = {
            'bitcoin': './training_data/bitcoin_5min.csv',
            'ethereum': './training_data/ethereum_5min.csv',
            'xau': './training_data/xau_5min.csv'
        }
        
        # Load pre-trained models
        self.load_models()
        
    def load_models(self):
        """Load pre-trained models for all assets"""
        print("Loading pre-trained models...")
        
        # Prepare data first (needed for feature columns)
        self.predictor.prepare_data(self.data_files)
        
        # Load models
        for asset_name in self.trading_pairs.keys():
            try:
                self.predictor.load_model(asset_name)
                print(f"✅ Loaded model for {asset_name}")
            except Exception as e:
                print(f"❌ Error loading model for {asset_name}: {e}")
    
    def get_current_price(self, trading_pair):
        """Get current price for a trading pair"""
        try:
            response = requests.get(self.current_price_url, params={
                "symbol": trading_pair,
                "resolution": "5",
                "from": int(time.time() - 300),
                "to": int(time.time())
            })
            data = response.json()
            
            price = data["c"][-1]
            if(price):
                return price
            else:
                print(f"No price data for {trading_pair}")
                return None
                
        except Exception as e:
            print(f"Error fetching price for {trading_pair}: {e}")
            return None
    
    def get_current_prices(self):
        """Get current prices for all assets"""
        current_prices = {}
        current_time = datetime.now(timezone.utc)
        
        # Round to the nearest minute (remove seconds)
        current_time = current_time.replace(second=0, microsecond=0)
        
        print(f"\n🕐 Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("Fetching current prices...")
        
        for asset_name, trading_pair in self.trading_pairs.items():
            price = self.get_current_price(trading_pair)
            if price:
                current_prices[asset_name] = {
                    'price': price,
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S')
                }
                print(f"💰 {asset_name}: ${price:,.2f}")
            else:
                print(f"❌ Failed to get price for {asset_name}")
        
        return current_prices, current_time
    
    def create_current_data_point(self, asset_name, price, timestamp):
        """Create a data point with current price and estimated OHLCV"""
        # For real-time, we'll use the current price for all OHLCV values
        # In a real implementation, you might want to get actual OHLCV data
        data_point = {
            'timestamp': timestamp,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': 0.0  # Volume might not be available in real-time
        }
        return data_point
    
    def prepare_prediction_data(self, asset_name, current_price, current_time):
        """Prepare data for prediction by combining historical data with current price, using a larger window and filling NaNs in the last row only (real-time best practice)"""
        try:
            # Load historical data
            df = pd.read_csv(self.data_files[asset_name])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Create current data point
            current_data = self.create_current_data_point(
                asset_name, current_price, current_time.strftime('%Y-%m-%d %H:%M:%S')
            )
            current_df = pd.DataFrame([current_data])
            current_df['timestamp'] = pd.to_datetime(current_df['timestamp'])
            # Combine and sort
            combined_df = pd.concat([df, current_df], ignore_index=True)
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            # Use a much larger window for feature creation
            window = self.predictor.sequence_length + 200
            combined_df = combined_df.tail(window)
            # Create features
            combined_df = self.predictor.preprocessor._create_features(combined_df)
            combined_df = combined_df.reset_index(drop=True)
            # Fill NaNs in the last row only (neutral values)
            last_idx = combined_df.index[-1]
            for col in combined_df.columns:
                if pd.isna(combined_df.at[last_idx, col]):
                    if 'rsi' in col:
                        combined_df.at[last_idx, col] = 50.0
                    elif 'macd' in col:
                        combined_df.at[last_idx, col] = 0.0
                    elif 'ma' in col or 'bb_' in col:
                        combined_df.at[last_idx, col] = combined_df.at[last_idx, 'close']
                    else:
                        combined_df.at[last_idx, col] = 0.0
            # Now drop NaNs in the rest
            if len(combined_df) > 1:
                combined_df = pd.concat([combined_df.iloc[:-1].dropna(), combined_df.iloc[[-1]]], ignore_index=True)
            else:
                combined_df = combined_df.dropna().reset_index(drop=True)
            print(f"After feature creation and NaN handling: {len(combined_df)} clean rows")
            return combined_df
        except Exception as e:
            print(f"Error preparing prediction data for {asset_name}: {e}")
            return None
    
    def predict_24h_forecast(self, asset_name, current_price, current_time):
        """Predict 24 hours of prices at 5-minute intervals"""
        try:
            # Prepare data
            df = self.prepare_prediction_data(asset_name, current_price, current_time)
            if df is None:
                return None
            
            # Get feature columns
            feature_columns = self.predictor.feature_columns[asset_name]
            
            # Scale data
            scaler = self.predictor.preprocessor.scalers[asset_name]
            scaled_data = scaler.transform(df[feature_columns])
            
            # Get last N data points for prediction
            seq_len = self.predictor.sequence_length
            if len(df) >= seq_len:
                recent_data = df.tail(seq_len)
                prediction_df = self.predictor.predict_next_24h(asset_name, recent_data)
                timestamps = prediction_df['timestamp'].tolist()
                predictions = prediction_df['predicted_price'].values
                return timestamps, predictions
            else:
                print(f"Insufficient data for {asset_name}: only {len(df)} clean rows after feature creation")
                return None
                
        except Exception as e:
            print(f"Error predicting for {asset_name}: {e}")
            return None
    
    def run_forecast(self):
        """Run the real-time forecasting system"""
        print("🚀 Starting Real-Time Price Forecasting System")
        print("=" * 60)
        
        while True:
            try:
                # Get current prices
                current_prices, current_time = self.get_current_prices()
                
                if not current_prices:
                    print("❌ No current prices available. Retrying in 60 seconds...")
                    time.sleep(60)
                    continue
                
                print(f"\n📊 Generating 24-hour forecasts...")
                print("=" * 60)
                
                # Generate forecasts for each asset
                for asset_name, price_data in current_prices.items():
                    print(f"\n🔮 Forecasting for {asset_name.upper()}")
                    print("-" * 40)
                    
                    result = self.predict_24h_forecast(
                        asset_name, 
                        price_data['price'], 
                        current_time
                    )
                    
                    if result:
                        timestamps, predictions = result
                        
                        # Display current price
                        print(f"Current Price: ${price_data['price']:,.2f}")
                        print(f"Current Time: {timestamps[0].strftime('%Y-%m-%d %H:%M:%S')} UTC")
                        
                        # Display first 10 predictions
                        print("\nFirst 10 predictions:")
                        for i in range(1, min(11, len(predictions))):
                            pred_time = timestamps[i].strftime('%H:%M:%S')
                            pred_price = predictions[i-1]
                            print(f"  {pred_time}: ${pred_price:,.2f}")
                        
                        # Display last 10 predictions
                        print("\nLast 10 predictions:")
                        for i in range(max(1, len(predictions)-9), len(predictions)):
                            pred_time = timestamps[i].strftime('%H:%M:%S')
                            pred_price = predictions[i-1]
                            print(f"  {pred_time}: ${pred_price:,.2f}")
                        
                        # Save forecast to file
                        self.save_forecast(asset_name, timestamps, predictions, current_time)
                        
                    else:
                        print(f"❌ Failed to generate forecast for {asset_name}")
                
                print(f"\n⏰ Next update in 60 seconds...")
                print("=" * 60)
                
                # Wait for 60 seconds before next update
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\n🛑 Forecasting stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in forecast loop: {e}")
                print("Retrying in 60 seconds...")
                time.sleep(60)
    
    def save_forecast(self, asset_name, timestamps, predictions, current_time):
        """Save forecast to file"""
        try:
            # Create forecasts directory
            os.makedirs("./forecasts", exist_ok=True)
            
            # Prepare data for saving
            forecast_data = []
            for i, (timestamp, prediction) in enumerate(zip(timestamps, predictions)):
                forecast_data.append({
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_price': prediction,
                    'is_current': i == 0
                })
            
            # Save to CSV
            df = pd.DataFrame(forecast_data)
            filename = f"./forecasts/{asset_name}_forecast_{current_time.strftime('%Y%m%d_%H%M')}.csv"
            df.to_csv(filename, index=False)
            
            print(f"💾 Forecast saved to: {filename}")
            
        except Exception as e:
            print(f"Error saving forecast: {e}")

def main():
    """Main function to run the real-time forecaster"""
    print("Real-Time Price Forecasting System")
    print("This system will:")
    print("- Fetch current prices every minute")
    print("- Predict 24 hours forward at 5-minute intervals")
    print("- Provide 289 price points (current + 288 future)")
    print("- Save forecasts to ./forecasts/ directory")
    print("\nPress Ctrl+C to stop")
    
    # Initialize forecaster
    forecaster = RealTimeForecaster()
    
    # Run the forecasting system
    forecaster.run_forecast()

if __name__ == "__main__":
    main()
