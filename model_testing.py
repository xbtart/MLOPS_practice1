# model_testing.py
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import os
import numpy as np

def test_model():
    model = joblib.load('model/model.pkl')
    
    for filename in os.listdir('data/test_preprocessed'):
        if filename.endswith('.csv'):
            data = pd.read_csv(os.path.join('data/test_preprocessed', filename))
            labels = np.random.randint(0, 2, size=data.shape[0])  # Используем те же случайные метки
            predictions = model.predict(data)
            accuracy = accuracy_score(labels, predictions)
            print(f'{filename}: Accuracy = {accuracy}')

if __name__ == "__main__":
    test_model()
