from core import PricePredictor

# Example usage
def main():
    # File paths for all trading pairs
    file_paths = {
        'bitcoin': './training_data/bitcoin_5min.csv',
        'ethereum': './training_data/ethereum_5min.csv',
        'xau': './training_data/xau_5min.csv'
    }
    
    # Initialize predictor
    predictor = PricePredictor(sequence_length=60, prediction_horizon=288)
    
    # Prepare data
    predictor.prepare_data(file_paths)
    
    # Train models for each asset
    for asset_name in file_paths.keys():
        print(f"\n{'='*50}")
        print(f"Training model for {asset_name}")
        print(f"{'='*50}")
        
        try:
            train_losses, val_losses = predictor.train_model(
                asset_name=asset_name,
                epochs=100,
                batch_size=32,
                lr=0.001
            )
            
            # Evaluate model
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
            print(f"Error training model for {asset_name}: {e}")
            continue

if __name__ == "__main__":
    main()