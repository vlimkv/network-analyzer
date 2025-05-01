import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 🔹 Загрузка данных
data = pd.read_csv("../data/processed/train_processed.csv")

# 🔹 Выбираем признаки (как у RF)
features = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment']
X = data[features].values
y = data['binary_label'].values

# 🔹 Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🔹 Разделение данных на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Преобразуем данные в формат [samples, timesteps, features] для LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 🔹 Создаем LSTM модель
model = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

# 🔹 Компиляция модели
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

# 🔹 Обучение модели
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# 🔹 Сохранение модели и скейлера
model.save("lstm_model.h5")
np.save("scaler_lstm.npy", scaler.mean_)

print("✅ LSTM модель успешно обучена и сохранена!")
