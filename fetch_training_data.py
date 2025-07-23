import numpy as np
from datetime import datetime, timedelta, timezone
import requests
import json
import time
import pandas as pd
import os

intervals = {'240m':'240', '60m':'60', '15m':'15', '5m':'5'}
start = '2022-01-01T00:00:00'
trading_pairs = ['Crypto.BTC/USD', 'Crypto.ETH/USD', 'Metal.XAU/USD']
BASE_URL = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
time_interval = 30 * 24 * 60 * 60  # 30 days in seconds

# Create training_data directory if it doesn't exist
os.makedirs("./training_data", exist_ok=True)

# Define file names for each trading pair
file_names = {
    'Crypto.BTC/USD': 'bitcoin_5min.csv',
    'Crypto.ETH/USD': 'ethereum_5min.csv', 
    'Metal.XAU/USD': 'xau_5min.csv'
}

# Prepare time range
start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
end_dt = datetime.now(timezone.utc)

def fetch_trading_pair_data(trading_pair):
    """Fetch data for a specific trading pair"""
    print(f"\n=== Fetching data for {trading_pair} ===")
    
    all_data = []
    current_start = start_dt
    
    while current_start < end_dt:
        current_end = min(current_start + timedelta(seconds=time_interval), end_dt)
        from_time = int(current_start.replace(tzinfo=timezone.utc).timestamp())
        to_time = int(current_end.replace(tzinfo=timezone.utc).timestamp())

        print(f"Fetching {trading_pair} from {current_start} to {current_end}")

        response = requests.get(BASE_URL, params={
            "symbol": trading_pair,
            "resolution": intervals['5m'],
            "from": from_time,
            "to": to_time
        })
        data = response.json()

        # Check for valid data
        if "s" in data and data["s"] == "ok" and "t" in data and data["t"]:
            # Process the new API response format
            timestamps = data["t"]
            opens = data["o"]
            highs = data["h"]
            lows = data["l"]
            closes = data["c"]
            volumes = data["v"]
            
            # Convert timestamps to datetime and create data points
            for i in range(len(timestamps)):
                timestamp = datetime.fromtimestamp(timestamps[i], tz=timezone.utc)
                data_point = {
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': opens[i],
                    'high': highs[i],
                    'low': lows[i],
                    'close': closes[i],
                    'volume': volumes[i]
                }
                all_data.append(data_point)
                
            print(f"Added {len(timestamps)} data points for {trading_pair}")
        else:
            print(f"No data for {trading_pair} from {current_start} to {current_end}")

        # Move to next window
        current_start = current_end
        time.sleep(1)  # Be polite to the API
    
    return all_data

# Fetch data for all trading pairs
for trading_pair in trading_pairs:
    try:
        # Fetch data for this trading pair
        pair_data = fetch_trading_pair_data(trading_pair)
        
        if pair_data:
            # Convert to DataFrame and save as CSV
            df = pd.DataFrame(pair_data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Save to CSV
            csv_filename = f"./training_data/{file_names[trading_pair]}"
            df.to_csv(csv_filename, index=False)
            
            print(f"✅ Saved {len(df)} data points to {csv_filename}")
            print(f"Data range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            
            # Save to npy for backward compatibility
            closes = np.array(df['close'].values, dtype=np.float32)
            times = np.array([datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp() for ts in df['timestamp']], dtype=np.int64)
            
            npy_filename = f"./training_data/{file_names[trading_pair].replace('.csv', '.npy')}"
            np.save(npy_filename, {"close": closes, "time": times})
            
        else:
            print(f"❌ No data fetched for {trading_pair}")
            
    except Exception as e:
        print(f"❌ Error fetching data for {trading_pair}: {e}")

print("\n=== Data Fetching Complete ===")
print("Files saved:")
for trading_pair, filename in file_names.items():
    filepath = f"./training_data/{filename}"
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"  ✅ {filename}: {len(df)} data points")
    else:
        print(f"  ❌ {filename}: Not created")