import threading
import time
import logging
import joblib
from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
from feature_extraction import FeatureExtractor
import pandas as pd
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealTimeDetector:
    def __init__(self, interface='eth0', model_path='models/ml_model.pkl'):
        self.interface = interface
        self.model = joblib.load(model_path)
        self.feature_extractor = FeatureExtractor()
        self.scaler = joblib.load('models/scaler.pkl')
        self.feature_columns = joblib.load('models/feature_columns.pkl')
        self.connection_dict = {}
        self.lock = threading.Lock()

    def packet_callback(self, packet):
        if IP in packet:
            try:
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

                # Create a unique connection ID
                connection_id = (src_ip, src_port, dst_ip, dst_port, protocol)

                with self.lock:
                    if connection_id not in self.connection_dict:
                        self.connection_dict[connection_id] = {
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
                            'urgent': 0,
                            'packet_count': 0
                        }

                    conn = self.connection_dict[connection_id]
                    conn['end_time'] = packet.time
                    if src_ip == conn['src_ip']:
                        conn['src_bytes'] += len(packet)
                    else:
                        conn['dst_bytes'] += len(packet)

                    # Save flags
                    if TCP in packet:
                        conn['flags'].add(packet[TCP].flags)

                    # Check for wrong fragments
                    if ip_layer.flags.DF == 0:
                        conn['wrong_fragment'] += 1

                    # Check for urgent packets
                    if TCP in packet and packet[TCP].urgptr > 0:
                        conn['urgent'] += 1

                    conn['packet_count'] += 1

                    # If connection is complete or timed out, process it
                    if conn['packet_count'] >= 5 or (conn['end_time'] - conn['start_time']) > 10:
                        self.process_connection(connection_id)
            except Exception as e:
                logger.error(f"Error processing packet: {e}")

    def process_connection(self, connection_id):
        conn = self.connection_dict.pop(connection_id, None)
        if conn:
            duration = conn['end_time'] - conn['start_time']
            protocol_type = self.feature_extractor.protocol_number_to_name(conn['protocol'])
            service = self.feature_extractor.port_to_service(conn['dst_port'])
            flag = self.feature_extractor.flags_to_str(conn['flags'])

            features = {
                'duration': duration,
                'src_bytes': conn['src_bytes'],
                'dst_bytes': conn['dst_bytes'],
                'land': conn['land'],
                'wrong_fragment': conn['wrong_fragment'],
                'urgent': conn['urgent'],
                # Add more features if required
            }

            # Since our model expects scaled numerical features, we need to scale them
            feature_df = pd.DataFrame([features])
            feature_df = feature_df.fillna(0)
            X = self.scaler.transform(feature_df)

            # Predict using the trained model
            prediction = self.model.predict(X)[0]

            # Map prediction to class labels
            label = 'Normal' if prediction == 0 else 'Anomalous'

            logger.info(f"Connection {connection_id} classified as {label}")
            print(f"🔎 Connection {connection_id} classified as {label}")

    def start_sniffing(self):
        logger.info(f"Starting packet sniffing on interface {self.interface}...")
        sniff(iface=self.interface, prn=self.packet_callback, store=0)

if __name__ == "__main__":
    interface = 'eth0'  # Change this to the appropriate network interface
    detector = RealTimeDetector(interface=interface)
    try:
        detector.start_sniffing()
    except KeyboardInterrupt:
        logger.info("Stopping packet sniffing.")
