import numpy as np
import pandas as pd
import tensorflow as tf

# 🔹 Загрузка модели и скейлера
model = tf.keras.models.load_model("lstm_model.h5")
scaler_mean = np.load("scaler_lstm.npy")

# 🔹 Симуляция нового сетевого трафика
new_data = pd.DataFrame({
    'duration': [5], 
    'src_bytes': [1000], 
    'dst_bytes': [2000], 
    'wrong_fragment': [0]
})

# 🔹 Нормализация
X_new = (new_data.values - scaler_mean) / np.std(new_data.values)

# 🔹 Преобразуем в формат LSTM
X_new = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))

# 🔹 Предсказание
prediction = model.predict(X_new)

# 🔹 Вывод результата
if prediction[0][0] > 0.5:
    print("⚠️ Внимание! Обнаружена потенциальная атака!")
else:
    print("✅ Трафик нормальный.")