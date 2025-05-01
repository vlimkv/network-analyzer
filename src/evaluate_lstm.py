import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Загрузка модели и скейлера
model = tf.keras.models.load_model("lstm_model.h5")
scaler_mean = np.load("scaler_lstm.npy")

# 🔹 Загрузка тестовых данных
data = pd.read_csv("../data/processed/train_processed.csv")
features = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment']
X_test = data[features].values
y_test = data['binary_label'].values

# 🔹 Нормализация
X_test = (X_test - scaler_mean) / np.std(X_test)

# 🔹 Преобразуем данные для LSTM
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 🔹 Предсказание
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# 🔹 Вывод метрик
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 🔹 Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('LSTM Confusion Matrix')
plt.show()
