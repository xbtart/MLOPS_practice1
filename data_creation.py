# data_creation.py
import numpy as np
import pandas as pd
import os

def create_dataset(n_samples=100, n_features=5, anomaly=False, noise_level=0.1):
    np.random.seed(42)
    data = np.random.rand(n_samples, n_features)
    
    # Добавление аномалий
    if anomaly:
        n_anomalies = int(0.1 * n_samples)
        anomalies = np.random.rand(n_anomalies, n_features) * 10
        data[:n_anomalies] = anomalies
    
    # Добавление шума
    noise = np.random.randn(n_samples, n_features) * noise_level
    data += noise
    
    return data

def save_datasets():
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    for i in range(5):
        train_data = create_dataset(anomaly=(i % 2 == 0), noise_level=0.1 * i)
        test_data = create_dataset(anomaly=(i % 2 == 1), noise_level=0.1 * i)
        
        pd.DataFrame(train_data).to_csv(f'data/train/dataset_{i}.csv', index=False)
        pd.DataFrame(test_data).to_csv(f'data/test/dataset_{i}.csv', index=False)

if __name__ == "__main__":
    save_datasets()
