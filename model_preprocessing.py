# model_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    scaler = StandardScaler()
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            data = pd.read_csv(os.path.join(input_dir, filename))
            scaled_data = scaler.fit_transform(data)
            pd.DataFrame(scaled_data).to_csv(os.path.join(output_dir, filename), index=False)

if __name__ == "__main__":
    preprocess_data('data/train', 'data/train_preprocessed')
    preprocess_data('data/test', 'data/test_preprocessed')
