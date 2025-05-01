class AlertingSystem:
    def __init__(self):
        pass  # Инициализация не требуется для вывода в консоль

    def send_alert(self, packet, is_anomaly, is_threat):
        print("\n=== Оповещение ===")
        print("Обнаружена подозрительная активность!")
        print(f"Детали пакета: {packet.summary()}")
        if is_anomaly:
            print("Тип: Аномальное поведение")
        if is_threat:
            print("Тип: Известная угроза")
        print("=================\n")
