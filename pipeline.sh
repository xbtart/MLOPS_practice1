#!/bin/bash

# Создание необходимых директорий
mkdir -p data/train data/test model

# Шаг 1: Создание данных
python ./data_creation.py

# Шаг 2: Предобработка данных
python ./model_preprocessing.py

# Шаг 3: Обучение модели
python ./model_preparation.py

# Шаг 4: Тестирование модели
python ./model_testing.py
