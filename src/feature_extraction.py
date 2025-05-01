import socket
import struct

import joblib
import pandas as pd
import numpy as np
from scapy.all import rdpcap
from scapy.layers.inet import  TCP, UDP, IP
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

class FeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.preprocessor = None

    def extract_features_from_pcap(self, pcap_file):
        """
        Извлекает признаки из PCAP файла и возвращает DataFrame с признаками
        """
        try:
            self.logger.info(f"Чтение PCAP файла: {pcap_file}")
            pcap_file = str(pcap_file)
            packets = rdpcap(pcap_file)
            self.logger.info(f"Количество прочитанных пакетов: {len(packets)}")

            # Словарь для хранения соединений
            connections = {}

            for packet in packets:
                if IP in packet:
                    ip_layer = packet[IP]
                    src_ip = ip_layer.src
                    dst_ip = ip_layer.dst
                    protocol = ip_layer.proto

                    if TCP in packet or UDP in packet:
                        if TCP in packet:
                            protocol_layer = packet[TCP]
                        else:
                            protocol_layer = packet[UDP]

                        src_port = protocol_layer.sport
                        dst_port = protocol_layer.dport
                    else:
                        src_port = 0
                        dst_port = 0

                    # Создаем уникальный идентификатор соединения
                    connection_id = (src_ip, src_port, dst_ip, dst_port, protocol)

                    # Инициализируем соединение в словаре, если его еще нет
                    if connection_id not in connections:
                        connections[connection_id] = {
                            'start_time': packet.time,
                            'end_time': packet.time,
                            'src_bytes': 0,
                            'dst_bytes': 0,
                            'protocol': protocol,
                            'src_ip': src_ip,
                            'dst_ip': dst_ip,
                            'src_port': src_port,
                            'dst_port': dst_port,
                            'flags': set(),
                            'land': int(src_ip == dst_ip and src_port == dst_port),
                            'wrong_fragment': 0,
                            'urgent': 0
                        }

                    # Обновляем информацию о соединении
                    conn = connections[connection_id]
                    conn['end_time'] = packet.time
                    if src_ip == conn['src_ip']:
                        conn['src_bytes'] += len(packet)
                    else:
                        conn['dst_bytes'] += len(packet)

                    # Сохраняем флаги
                    if TCP in packet:
                        conn['flags'].add(packet[TCP].flags)

                    # Проверяем на некорректные фрагменты
                    if ip_layer.flags.DF == 0:
                        conn['wrong_fragment'] += 1

                    # Проверяем на срочные пакеты
                    if TCP in packet and packet[TCP].urgptr > 0:
                        conn['urgent'] += 1

            # Преобразуем словарь соединений в список признаков
            features_list = []
            for conn_id, conn in connections.items():
                duration = conn['end_time'] - conn['start_time']
                protocol_type = self.protocol_number_to_name(conn['protocol'])
                service = self.port_to_service(conn['dst_port'])
                flag = self.flags_to_str(conn['flags'])

                features = {
                    'duration': duration,
                    'protocol_type': protocol_type,
                    'service': service,
                    'flag': flag,
                    'src_bytes': conn['src_bytes'],
                    'dst_bytes': conn['dst_bytes'],
                    'land': conn['land'],
                    'wrong_fragment': conn['wrong_fragment'],
                    'urgent': conn['urgent'],
                    # Добавьте дополнительные признаки по необходимости
                }

                features_list.append(features)

            # Преобразуем в DataFrame
            df = pd.DataFrame(features_list)
            self.logger.info(f"Извлечено {df.shape[0]} соединений с признаками")
            return df

        except Exception as e:
            self.logger.error(f"Ошибка при извлечении признаков из PCAP файла: {str(e)}")
            raise

    def protocol_number_to_name(self, proto_number):
        """
        Преобразует номер протокола в его название
        """
        proto_map = {6: 'tcp', 17: 'udp', 1: 'icmp'}
        return proto_map.get(proto_number, 'other')

    def port_to_service(self, port):
        """
        Преобразует номер порта в название сервиса
        """
        # Для простоты используем ограниченный список
        service_map = {
            80: 'http',
            21: 'ftp',
            22: 'ssh',
            23: 'telnet',
            25: 'smtp',
            53: 'dns',
            443: 'https',
            # Добавьте дополнительные порты и сервисы по необходимости
        }
        return service_map.get(port, 'other')

    def flags_to_str(self, flags_set):
        """
        Преобразует множество флагов TCP в строковое представление
        """
        if not flags_set:
            return 'OTH'
        flag_str = ''.join([str(flag) for flag in flags_set])
        return flag_str

    def extract_features(self, X, fit=False):
        """
        Extracts and scales features from the input DataFrame X.

        Args:
            X (DataFrame): Input features DataFrame.
            fit (bool): If True, fit the scaler on X. Used for training data.

        Returns:
            DataFrame: Scaled features DataFrame.
        """
        numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
        X_numeric = X[numeric_columns]

        # Handle missing values
        X_numeric = X_numeric.fillna(0)

        if fit:
            # Fit the scaler on training data
            X_scaled = self.scaler.fit_transform(X_numeric)
        else:
            # Transform using the existing scaler
            X_scaled = self.scaler.transform(X_numeric)

        # Return DataFrame with scaled features
        X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_columns)
        return X_scaled_df

    def fit_transform(self, X):
        """
        Fits the preprocessor on X and transforms X.
        Returns transformed features as a numpy array.
        """
        try:
            # Define numerical and categorical columns
            numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_columns = list(set(X.columns) - set(numeric_columns))

            # Handle missing values
            X = X.fillna(0)

            # Create preprocessing pipelines for numeric and categorical data
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')

            # Create a column transformer
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_columns),
                    ('cat', categorical_transformer, categorical_columns)
                ]
            )

            # Fit and transform the data
            X_transformed = self.preprocessor.fit_transform(X)

            # Save the preprocessor for later use
            joblib.dump(self.preprocessor, 'models/preprocessor.pkl')

            return X_transformed

        except Exception as e:
            self.logger.error(f"Error in feature extraction: {e}")
            raise

    def transform(self, X):
        """
        Transforms X using the pre-fitted preprocessor.
        Returns transformed features as a numpy array.
        """
        try:
            if self.preprocessor is None:
                # Load the preprocessor
                self.preprocessor = joblib.load('models/preprocessor.pkl')

            # Handle missing values
            X = X.fillna(0)

            # Transform the data
            X_transformed = self.preprocessor.transform(X)

            return X_transformed

        except Exception as e:
            self.logger.error(f"Error in feature transformation: {e}")
            raise