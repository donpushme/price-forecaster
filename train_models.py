import os
import pandas as pd
import numpy as np
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
                print(f"   Available columns: {list(df.columns)}")
                continue
            
            # Check if data is not empty
            if len(df) == 0:
                print(f"❌ File is empty: {file_path}")
                continue
            
            # Convert timestamp and check for valid dates
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if df['timestamp'].isna().all():
                    print(f"❌ Invalid timestamp data in: {file_path}")
                    continue
            except Exception as e:
                print(f"❌ Error parsing timestamps in {file_path}: {e}")
                continue
            
            # Check for reasonable price data
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if df[col].isna().all() or (df[col] <= 0).all():
                    print(f"❌ Invalid {col} data in: {file_path}")
                    continue
            
            # Check data continuity (look for large gaps)
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff()
            expected_interval = pd.Timedelta(minutes=5)
            large_gaps = time_diffs > expected_interval * 2  # Allow some tolerance
            
            if large_gaps.sum() > len(df) * 0.1:  # If more than 10% have large gaps
                print(f"⚠️  Warning: Found {large_gaps.sum()} large time gaps in {file_path}")
                print("   This may affect model performance")
            
            print(f"✅ Valid data file: {len(df)} records")
            print(f"   Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
            print(f"   Data completeness: {(1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")
            
            # Estimate minimum required data for training
            min_required = 60 + 288 + 1000  # sequence_length + prediction_horizon + minimum training samples
            if len(df) < min_required:
                print(f"⚠️  Warning: Only {len(df)} records. Recommended minimum: {min_required}")
                print("   Model performance may be limited with insufficient data")
            
            valid_files[asset_name] = file_path
            
        except Exception as e:
            print(f"❌ Error reading file {file_path}: {e}")
            continue
    
    return valid_files

def validate_training_data(predictor, asset_name):
    """Validate that training data is sufficient"""
    if asset_name not in predictor.sequences:
        return False, "No sequences created"
    
    sequences = predictor.sequences[asset_name]
    targets = predictor.targets[asset_name]
    
    # Check minimum data requirements
    min_sequences = 1000  # Minimum for meaningful training
    recommended_sequences = 5000  # Recommended for good performance
    
    if len(sequences) < min_sequences:
        return False, f"Insufficient data: {len(sequences)} sequences (minimum: {min_sequences})"
    
    if len(sequences) < recommended_sequences:
        return True, f"Limited data: {len(sequences)} sequences (recommended: {recommended_sequences})"
    
    return True, f"Good data: {len(sequences)} sequences"

def main():
    """Main function to train models with improved error handling"""
    print("🚀 Advanced Price Forecasting Model Training")
    print("=" * 60)
    
    # File paths for all trading pairs
    file_paths = {
        'bitcoin': './training_data/bitcoin_5min.csv',
        'ethereum': './training_data/ethereum_5min.csv',
        'xau': './training_data/xau_5min.csv'
    }
    
    # Check data files
    print("📋 Checking data files...")
    valid_files = check_data_files(file_paths)
    
    if not valid_files:
        print("\n❌ No valid data files found!")
        print("Please ensure your CSV files are in the correct format with columns:")
        print("   timestamp, open, high, low, close, volume")
        print("And run the data collection script to gather training data.")
        return
    
    print(f"\n✅ Found {len(valid_files)} valid data files")
    
    # Initialize predictor with improved parameters
    print("\n🔧 Initializing PricePredictor...")
    predictor = PricePredictor(
        sequence_length=120,    # 10 hours of context (improved from 60)
        prediction_horizon=288  # 24 hours prediction
    )
    
    # Prepare data only for valid files
    print("📊 Preparing and preprocessing data...")
    try:
        predictor.prepare_data(valid_files)
        print("✅ Data preparation completed successfully")
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        print("Please check your CSV file format and try again.")
        return
    
    # Validate data for each asset before training
    valid_assets_for_training = []
    
    for asset_name in valid_files.keys():
        is_valid, message = validate_training_data(predictor, asset_name)
        print(f"\n📈 {asset_name}: {message}")
        
        if is_valid:
            valid_assets_for_training.append(asset_name)
        else:
            print(f"⚠️  Skipping {asset_name} training due to insufficient data")
    
    if not valid_assets_for_training:
        print("\n❌ No assets have sufficient data for training!")
        return
    
    # Train models for each valid asset
    trained_models = {}
    
    for asset_name in valid_assets_for_training:
        print(f"\n{'='*60}")
        print(f"🔥 Training model for {asset_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Display training configuration
            sequences = predictor.sequences[asset_name]
            train_size = int(len(sequences) * 0.7)
            val_size = int(len(sequences) * 0.2)
            test_size = len(sequences) - train_size - val_size
            
            print(f"📊 Training Configuration:")
            print(f"   Total sequences: {len(sequences)}")
            print(f"   Training set: {train_size} sequences")
            print(f"   Validation set: {val_size} sequences")
            print(f"   Test set: {test_size} sequences")
            print(f"   Sequence length: 120 (10 hours)")
            print(f"   Prediction horizon: 288 (24 hours)")
            
            # Train with improved parameters
            train_losses, val_losses = predictor.train_model(
                asset_name=asset_name,
                epochs=200,        # Increased from 100
                batch_size=64,     # Increased from 32
                lr=0.0005         # Decreased from 0.001
            )
            
            print(f"\n📈 Training completed for {asset_name}")
            print(f"   Final training loss: {train_losses[-1]:.6f}")
            print(f"   Final validation loss: {val_losses[-1]:.6f}")
            
            # Evaluate model performance
            print(f"\n🎯 Evaluating model for {asset_name}...")
            try:
                mae, rmse = predictor.evaluate_model(asset_name)
                trained_models[asset_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'train_loss': train_losses[-1],
                    'val_loss': val_losses[-1]
                }
            except Exception as e:
                print(f"⚠️  Error evaluating {asset_name}: {e}")
            
            # Make a sample prediction to test the model
            print(f"\n🔮 Testing prediction for {asset_name}...")
            try:
                # Get the last 120 data points for prediction
                last_data = predictor.data[asset_name].tail(120)
                
                if len(last_data) >= 120:
                    prediction_df = predictor.predict_next_24h(asset_name, last_data)
                    
                    # Display prediction summary
                    current_price = last_data['close'].iloc[-1]
                    predicted_prices = prediction_df['predicted_price'].values
                    
                    print(f"✅ Sample prediction successful:")
                    print(f"   Current price: ${current_price:,.2f}")
                    print(f"   1-hour ahead: ${predicted_prices[11]:,.2f}")
                    print(f"   6-hour ahead: ${predicted_prices[71]:,.2f}")
                    print(f"   24-hour ahead: ${predicted_prices[-1]:,.2f}")
                    
                    change_24h = ((predicted_prices[-1] - current_price) / current_price) * 100
                    print(f"   24h predicted change: {change_24h:+.2f}%")
                    
                else:
                    print(f"⚠️  Insufficient recent data for prediction test")
                    
            except Exception as e:
                print(f"⚠️  Error testing prediction for {asset_name}: {e}")
                
        except Exception as e:
            print(f"❌ Error training model for {asset_name}: {e}")
            print(f"   Skipping {asset_name} and continuing with other assets...")
            continue
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 TRAINING SUMMARY")
    print(f"{'='*60}")
    
    if trained_models:
        print(f"✅ Successfully trained {len(trained_models)} models:")
        
        for asset_name, metrics in trained_models.items():
            print(f"\n📊 {asset_name.upper()}:")
            print(f"   MAE: ${metrics['mae']:,.2f}")
            print(f"   RMSE: ${metrics['rmse']:,.2f}")
            print(f"   Training Loss: {metrics['train_loss']:.6f}")
            print(f"   Validation Loss: {metrics['val_loss']:.6f}")
        
        print(f"\n💾 Models saved in: {predictor.models_dir}")
        print("📁 Model files:")
        for asset_name in trained_models.keys():
            model_file = f"best_model_{asset_name}.pth"
            print(f"   - {model_file}")
        
        print("\n🚀 Next steps:")
        print("   1. Run real-time prediction script to use the trained models")
        print("   2. Monitor model performance over time")
        print("   3. Retrain periodically with new data")
        
    else:
        print("❌ No models were successfully trained!")
        print("Please check your data quality and try again.")
    
    print(f"\n{'='*60}")
    print("Training process completed!")

if __name__ == "__main__":
    main()