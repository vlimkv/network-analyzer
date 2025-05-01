import pandas as pd
import joblib

# 🔹 Загрузка модели
rf_model = joblib.load("rf_model_fixed.pkl")

# 🔹 Файл для логов
log_file = "attack_logs.txt"

# 🔹 Симуляция сетевого трафика
new_data = pd.DataFrame({
    'duration': [2, 30, 200], 
    'src_bytes': [200, 10000, 50000], 
    'dst_bytes': [5000, 3000, 7000], 
    'wrong_fragment': [0, 1, 0]
})

# 🔹 Предсказание модели
predictions = rf_model.predict(new_data)

# 🔹 Запись результатов в лог-файл (без эмодзи)
with open(log_file, "a", encoding="utf-8") as log:
    for i, pred in enumerate(predictions):
        if pred == 1:
            log.write(f"Атака обнаружена! Запись {i} \n")
        else:
            log.write(f"Нормальный трафик: запись {i} \n")

print(f"Лог атак записан в {log_file}")
