# model_preparation.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np

def train_model():
    all_data = []
    all_labels = []
    
    for filename in os.listdir('data/train_preprocessed'):
        if filename.endswith('.csv'):
            data = pd.read_csv(os.path.join('data/train_preprocessed', filename))
            labels = np.random.randint(0, 2, size=data.shape[0])  # Для примера генерируем случайные метки
            all_data.append(data)
            all_labels.append(labels)
    
    X_train = pd.concat(all_data)
    y_train = np.hstack(all_labels)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/model.pkl')

if __name__ == "__main__":
    train_model()
