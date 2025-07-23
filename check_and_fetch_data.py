import os
import subprocess
import sys

def check_missing_data():
    """Check which data files are missing"""
    required_files = {
        'bitcoin': './training_data/bitcoin_5min.csv',
        'ethereum': './training_data/ethereum_5min.csv',
        'xau': './training_data/xau_5min.csv'
    }
    
    missing_files = []
    existing_files = []
    
    print("Checking data files...")
    print("=" * 40)
    
    for asset_name, file_path in required_files.items():
        if os.path.exists(file_path):
            # Check if file has content
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Has header + at least one data row
                        existing_files.append(asset_name)
                        print(f"✅ {asset_name}: {file_path}")
                    else:
                        missing_files.append(asset_name)
                        print(f"❌ {asset_name}: {file_path} (empty file)")
            except Exception as e:
                missing_files.append(asset_name)
                print(f"❌ {asset_name}: {file_path} (error reading: {e})")
        else:
            missing_files.append(asset_name)
            print(f"❌ {asset_name}: {file_path} (file not found)")
    
    print(f"\nSummary:")
    print(f"✅ Existing files: {len(existing_files)}")
    print(f"❌ Missing files: {len(missing_files)}")
    
    if missing_files:
        print(f"\nMissing files: {', '.join(missing_files)}")
        return missing_files
    else:
        print("\n🎉 All data files are present and valid!")
        return []

def fetch_missing_data():
    """Fetch missing data using fetch_training_data.py"""
    print("\nFetching missing data...")
    print("=" * 40)
    
    try:
        # Run the fetch script
        result = subprocess.run([sys.executable, 'fetch_training_data.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Data fetching completed successfully!")
            print("Output:")
            print(result.stdout)
        else:
            print("❌ Error fetching data:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Data fetching timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running fetch script: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("Data File Checker and Fetcher")
    print("=" * 40)
    
    # Check for missing files
    missing_files = check_missing_data()
    
    if missing_files:
        print(f"\nWould you like to fetch the missing data for: {', '.join(missing_files)}?")
        response = input("Enter 'y' to continue or any other key to exit: ")
        
        if response.lower() == 'y':
            success = fetch_missing_data()
            if success:
                print("\nRe-checking data files...")
                check_missing_data()
            else:
                print("\nFailed to fetch data. Please run 'python fetch_training_data.py' manually.")
        else:
            print("Skipping data fetch.")
    else:
        print("\nAll data files are ready! You can now run:")
        print("python train_models.py")

if __name__ == "__main__":
    main() 