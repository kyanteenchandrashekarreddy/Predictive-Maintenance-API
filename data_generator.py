import pandas as pd
import numpy as np
import random

def generate_synthetic_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    # Base signals
    # Temperature: Gaussian centered at 60, std 10
    temperature = np.random.normal(60, 10, n_samples)
    
    # Vibration: Exponential distribution
    vibration = np.random.exponential(scale=2, size=n_samples)
    
    # Pressure: Uniform 100-300
    pressure = np.random.uniform(100, 300, n_samples)

    # DataFrame
    df = pd.DataFrame({
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure
    })

    # Target: Machine Failure
    # Logic: High temp OR High vibration -> higher chance of failure
    # We create a probability score
    prob_fail = (
        (df['temperature'] > 80).astype(int) * 0.4 +
        (df['vibration'] > 5).astype(int) * 0.4 +
        (df['pressure'] < 120).astype(int) * 0.1
    )
    # Add random noise
    prob_fail += np.random.uniform(0, 0.2, n_samples)
    
    # Threshold for failure
    df['machine_failure'] = (prob_fail > 0.5).astype(int)

    # --- Injecting Problems ---

    # 1. Missing Values (NaN)
    # Randomly set 5% of temperature to NaN
    nan_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[nan_indices, 'temperature'] = np.nan

    # 2. Outliers
    # Introduce extreme spikes in vibration for 2% of data
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    df.loc[outlier_indices, 'vibration'] = df.loc[outlier_indices, 'vibration'] * 10

    # 3. Mismatched Data Types
    # Convert some pressure values to strings (simulating sensor error "ERR")
    # Actually, let's make it a numeric string to be tricky, or just a string "Low"
    type_err_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    # Pandas object column if we insert strings
    df['pressure'] = df['pressure'].astype(object) 
    df.loc[type_err_indices, 'pressure'] = "ERR"

    print(f"Generated {n_samples} samples.")
    print("Injected 5% missing temps, 2% vibration outliers, 1% pressure type errors.")
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("raw_data.csv", index=False)
    print("Saved to raw_data.csv")
