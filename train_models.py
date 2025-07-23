import os
import pandas as pd
from core import PricePredictor

def check_data_files(file_paths):
    """Check if all required data files exist and have valid data"""
    valid_files = {}
    
    for asset_name, file_path in file_paths.items():
        print(f"\nChecking {asset_name} data file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        
        try:
            # Load and validate the CSV file
            df = pd.read_csv(file_path)
            
            # Check required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                continue
            
            # Check if data is not empty
            if len(df) == 0:
                print(f"❌ File is empty: {file_path}")
                continue
            
            # Convert timestamp and check for valid dates
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['timestamp'].isna().all():
                print(f"❌ Invalid timestamp data in: {file_path}")
                continue
            
            print(f"✅ Valid data file: {len(df)} records")
            print(f"   Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
            
            valid_files[asset_name] = file_path
            
        except Exception as e:
            print(f"❌ Error reading file {file_path}: {e}")
            continue
    
    return valid_files

def main():
    """Main function to train models"""
    print("🚀 Price Forecasting Model Training")
    print("=" * 50)
    
    # File paths for all trading pairs
    file_paths = {
        'bitcoin': './training_data/bitcoin_5min.csv',
        'ethereum': './training_data/ethereum_5min.csv',
        'xau': './training_data/xau_5min.csv'
    }
    
    # Check data files
    print("Checking data files...")
    valid_files = check_data_files(file_paths)
    
    if not valid_files:
        print("\n❌ No valid data files found!")
        print("Please run 'python fetch_training_data.py' to fetch the required data.")
        return
    
    print(f"\n✅ Found {len(valid_files)} valid data files")
    
    # Initialize predictor
    print("\nInitializing PricePredictor...")
    predictor = PricePredictor(sequence_length=60, prediction_horizon=288)
    
    # Prepare data only for valid files
    print("Preparing data...")
    try:
        predictor.prepare_data(valid_files)
        print("✅ Data preparation completed")
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return
    
    # Train models for each valid asset
    for asset_name in valid_files.keys():
        print(f"\n{'='*50}")
        print(f"Training model for {asset_name}")
        print(f"{'='*50}")
        
        try:
            # Check if we have enough data for training
            sequences = predictor.sequences[asset_name]
            if len(sequences) < 100:
                print(f"⚠️  Warning: Only {len(sequences)} sequences available for {asset_name}")
                print("   This might not be enough for effective training.")
            
            train_losses, val_losses = predictor.train_model(
                asset_name=asset_name,
                epochs=100,
                batch_size=32,
                lr=0.001
            )
            
            # Evaluate model
            print(f"\nEvaluating model for {asset_name}...")
            predictor.evaluate_model(asset_name)
            
            # Example prediction using the last 60 data points from the dataset
            print(f"\nMaking sample prediction for {asset_name}...")
            last_60_data = predictor.data[asset_name].tail(60).to_dict('records')
            
            try:
                prediction = predictor.predict_next_24h(asset_name, last_60_data)
                print(f"{asset_name} 24h prediction (next 288 5-min intervals):")
                print(f"First 10 predictions: {prediction[:10]}")
                print(f"Last 10 predictions: {prediction[-10:]}")
            except Exception as e:
                print(f"Prediction error for {asset_name}: {e}")
                
        except Exception as e:
            print(f"❌ Error training model for {asset_name}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("🎉 Training completed!")
    print("Models saved in ./models/ directory")
    print("You can now run 'python real_time_forecast.py' for real-time predictions")

if __name__ == "__main__":
    main()