import pandas as pd
import joblib

# 🔹 Загрузка модели
rf_model = joblib.load("rf_model_fixed.pkl")

# 🔹 Симуляция входящих данных (или получи их из сети)
new_data = pd.DataFrame({
    'duration': [5], 
    'src_bytes': [1000], 
    'dst_bytes': [2000], 
    'wrong_fragment': [0]
})

# 🔹 Предсказание модели
prediction = rf_model.predict(new_data)

# 🔹 Вывод результата
if prediction[0] == 1:
    print("⚠️ Внимание! Обнаружена потенциальная атака!")
else:
    print("✅ Трафик нормальный.")
