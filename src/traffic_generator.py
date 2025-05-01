import pandas as pd
import random
from pathlib import Path
from scapy.all import Raw, wrpcap
from scapy.layers.inet import IP, TCP, UDP
import logging
import time

class TrafficGenerator:
    def __init__(self, output_dir="data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Configure logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Define common services and their ports
        self.services = {
            'http': 80,
            'ftp': 21,
            'ssh': 22,
            'telnet': 23,
            'smtp': 25,
            'dns': 53
        }

        # Features common between PCAP, KDD, and UNSW datasets
        self.common_features = [
            'duration',          # Handled during feature extraction
            'protocol_type',     # Derived from IP proto field
            'service',           # Based on port numbers
            'flag',              # TCP flags
            'src_bytes',         # Calculated during feature extraction
            'dst_bytes',         # Calculated during feature extraction
            'land',              # Determined if src and dst IP and ports are the same
            'wrong_fragment',    # Not directly generated here
            'urgent'             # TCP urgent flag
        ]

    def generate_normal_packet(self):
        """Generate normal network traffic packet"""
        src_ip = f"192.168.1.{random.randint(2, 254)}"
        dst_ip = f"192.168.1.{random.randint(2, 254)}"
        while dst_ip == src_ip:
            dst_ip = f"192.168.1.{random.randint(2, 254)}"

        # Select random service
        service = random.choice(list(self.services.keys()))
        dst_port = self.services[service]
        src_port = random.randint(1024, 65535)

        # Choose protocol based on service
        if service == 'dns':
            proto = UDP
        else:
            proto = TCP

        # Create packet with realistic size and flags
        ip = IP(src=src_ip, dst=dst_ip)
        if proto == TCP:
            tcp_flags = random.choice(['S', 'A', 'PA', 'FA'])
            layer4 = TCP(sport=src_port, dport=dst_port, flags=tcp_flags)
        else:
            layer4 = UDP(sport=src_port, dport=dst_port)

        # Add random payload
        payload_size = random.randint(64, 1024)
        payload = Raw(load='X' * payload_size)

        return ip / layer4 / payload

    def generate_attack_packet(self):
        """Generate attack traffic packet"""
        attack_types = ['syn_flood', 'port_scan']
        attack = random.choice(attack_types)

        src_ip = f"192.168.1.{random.randint(2, 254)}"
        dst_ip = f"192.168.1.{random.randint(2, 254)}"
        if attack == 'syn_flood':
            # SYN Flood attack
            ip = IP(src=src_ip, dst=dst_ip)
            tcp = TCP(sport=random.randint(1024, 65535),
                      dport=random.choice(list(self.services.values())),
                      flags='S')
            payload = Raw(load='X' * random.randint(64, 512))
            return ip / tcp / payload
        elif attack == 'port_scan':
            # Port scanning
            packets = []
            for port in range(20, 1024, random.randint(10, 50)):
                ip = IP(src=src_ip, dst=dst_ip)
                tcp = TCP(sport=random.randint(1024, 65535),
                          dport=port,
                          flags='S')
                payload = Raw(load='X' * random.randint(64, 512))
                packets.append(ip / tcp / payload)
            return packets

    def generate_traffic_flow(self, total_packets=1000, attack_ratio=0.2, duration=600):
        """Generate traffic flow with a specified number of packets and attack ratio"""
        packets = []
        labels = []
        timestamps = []

        self.logger.info(f"Generating {total_packets} packets with attack ratio {attack_ratio}")

        # Simulate timestamps over the duration period
        start_time = time.time()
        time_stamps = sorted([random.uniform(start_time, start_time + duration) for _ in range(total_packets)])

        for i in range(total_packets):
            is_attack = random.random() < attack_ratio

            if is_attack:
                attack_packet = self.generate_attack_packet()
                if isinstance(attack_packet, list):
                    packets.extend(attack_packet)
                    labels.extend([1] * len(attack_packet))  # Attack labels
                    timestamps.extend([time_stamps[i]] * len(attack_packet))
                else:
                    packets.append(attack_packet)
                    labels.append(1)  # Attack label
                    timestamps.append(time_stamps[i])
            else:
                packet = self.generate_normal_packet()
                packets.append(packet)
                labels.append(0)  # Normal label
                timestamps.append(time_stamps[i])

        return packets, labels, timestamps

    def generate_and_save(self, filename="sample.pcap", total_packets=1000, duration=600):
        """Generate and save traffic to PCAP with labels"""
        self.logger.info(f"Starting traffic generation for {filename}")

        # Generate mixed traffic
        packets, labels, timestamps = self.generate_traffic_flow(
            total_packets=total_packets,
            attack_ratio=0.2,
            duration=duration
        )

        # Save packets to PCAP
        pcap_path = self.output_dir / filename
        wrpcap(str(pcap_path), packets)

        # Save labels
        labels_df = pd.DataFrame({
            'packet_id': range(len(labels)),
            'timestamp': timestamps,
            'binary_label': labels
        })

        labels_path = self.output_dir / f"{filename}_labels.csv"
        labels_df.to_csv(labels_path, index=False)

        self.logger.info(f"Generated {len(packets)} packets")
        self.logger.info(f"Attack ratio: {sum(labels) / len(labels):.2%}")
        self.logger.info(f"Files saved to {self.output_dir}")

        return packets, labels

if __name__ == "__main__":
    generator = TrafficGenerator()
    packets, labels = generator.generate_and_save(
        filename="sample.pcap",
        total_packets=100000,
        duration=600  # Simulate 10 minutes of traffic
    )

    print(f"\nTraffic Generation Summary:")
    print(f"Total packets: {len(packets)}")
    print(f"Attack packets: {sum(labels)}")
    print(f"Normal packets: {len(labels) - sum(labels)}")
    print(f"Attack ratio: {sum(labels) / len(labels):.2%}")
