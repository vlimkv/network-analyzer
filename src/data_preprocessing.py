import os
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from feature_extraction import FeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_dir='/media/weiss/Other/MDIR_USR/Scripts/Disser/system/data', processed_dir='/media/weiss/Other/MDIR_USR/Scripts/Disser/system/data/processed'):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        self.feature_extractor = FeatureExtractor()

        # Define common features between datasets
        self.common_features = [
            'duration',          # KDD, UNSW, PCAP
            'protocol_type',     # KDD (protocol_type), UNSW (proto)
            'service',           # KDD, UNSW
            'flag',              # KDD (flag), UNSW (state)
            'src_bytes',         # KDD, UNSW (sbytes)
            'dst_bytes',         # KDD, UNSW (dbytes)
            'land',              # KDD, UNSW (is_sm_ips_ports)
            'wrong_fragment',    # KDD, UNSW (we may need to set default)
            'urgent'             # KDD, UNSW (urgent)
        ]

    def load_kdd_dataset(self, file_path):
        logger.info(f"Loading KDD dataset: {file_path}")
        try:
            # KDD dataset has 42 or 43 columns
            df = pd.read_csv(file_path, header=None)
            # Assign column names
            column_names = [
                'duration', 'protocol_type', 'service', 'flag',
                'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
                # ... (other columns)
                # The last column is 'label'
            ]
            df = df.iloc[:, :len(column_names)+1]  # Ensure we have at least the columns we need
            df.columns = column_names + ['label']
            return df
        except Exception as e:
            logger.error(f"Error loading KDD dataset: {e}")
            raise

    def preprocess_kdd(self, df):
        logger.info("Preprocessing KDD dataset")
        try:
            # Assign binary labels: 'normal' -> 0, others -> 1
            df['binary_label'] = df['label'].apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)

            # Select only common features
            df = df[self.common_features + ['binary_label']]

            # Log class distribution
            logger.info(f"KDD Class distribution:\n{df['binary_label'].value_counts()}")

            return df
        except Exception as e:
            logger.error(f"Error preprocessing KDD dataset: {e}")
            raise

    def load_unsw_dataset(self, file_path):
        logger.info(f"Loading UNSW-NB15 dataset: {file_path}")
        try:
            # Define UNSW-NB15 column names
            unsw_columns = [
                'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
                'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
                'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
                'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
                'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
                'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
                'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
                'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label'
            ]

            # Read CSV with specified column names and handle mixed types
            df = pd.read_csv(file_path, names=unsw_columns, low_memory=False)
            return df
        except Exception as e:
            logger.error(f"Error loading UNSW-NB15 dataset: {e}")
            raise

    def preprocess_unsw(self, df):
        logger.info("Preprocessing UNSW-NB15 dataset")
        try:
            # Map UNSW columns to common features
            column_mapping = {
                'dur': 'duration',
                'proto': 'protocol_type',
                'service': 'service',
                'state': 'flag',
                'sbytes': 'src_bytes',
                'dbytes': 'dst_bytes',
                'is_sm_ips_ports': 'land',
                'Label': 'binary_label'
            }
            df = df.rename(columns=column_mapping)

            # Add missing features with default values
            df['wrong_fragment'] = 0  # UNSW doesn't have this feature
            df['urgent'] = 0  # UNSW doesn't have this feature

            # Ensure binary_label is correctly set
            df['binary_label'] = df['binary_label'].astype(int)

            # Log class distribution
            logger.info(f"UNSW Class distribution:\n{df['binary_label'].value_counts()}")

            return df[self.common_features + ['binary_label']]
        except Exception as e:
            logger.error(f"Error preprocessing UNSW dataset: {e}")
            raise

    def process_pcap(self, pcap_file):
        logger.info(f"Processing PCAP file: {pcap_file}")
        try:
            features_df = self.feature_extractor.extract_features_from_pcap(pcap_file)
            # Since we may not have labels, set binary_label to None or 0 by default
            features_df['binary_label'] = 0  # Or load actual labels if available
            # Select only common features
            features_df = features_df[self.common_features + ['binary_label']]
            return features_df
        except Exception as e:
            logger.error(f"Error processing PCAP file: {e}")
            raise

    def align_features(self, dfs):
        logger.info("Aligning features across datasets")
        try:
            # Get all the columns
            all_columns = set()
            for df in dfs:
                all_columns.update(df.columns)

            # Ensure all datasets have the same columns
            for i in range(len(dfs)):
                missing_cols = all_columns - set(dfs[i].columns)
                for col in missing_cols:
                    dfs[i][col] = 0  # Fill missing columns with default value
                # Reorder columns
                dfs[i] = dfs[i][self.common_features + ['binary_label']]

            return dfs
        except Exception as e:
            logger.error(f"Error aligning features: {e}")
            raise

    def get_combined_dataset(self):
        logger.info("Loading and combining datasets")

        # Load and preprocess KDD datasets
        kdd_train_path = self.data_dir / 'KDDTrain+.txt'
        kdd_test_path = self.data_dir / 'KDDTest+.txt'

        df_train_kdd = self.load_kdd_dataset(kdd_train_path)
        df_train_kdd = self.preprocess_kdd(df_train_kdd)

        df_test_kdd = self.load_kdd_dataset(kdd_test_path)
        df_test_kdd = self.preprocess_kdd(df_test_kdd)

        # Load and preprocess UNSW-NB15 dataset
        unsw_path = self.data_dir / 'UNSW-NB15_1.csv'
        df_unsw = self.load_unsw_dataset(unsw_path)
        df_unsw = self.preprocess_unsw(df_unsw)

        # Combine datasets
        df_combined = pd.concat([df_train_kdd, df_test_kdd, df_unsw], ignore_index=True)

        # Align features
        df_combined_list = self.align_features([df_combined])
        df_combined = df_combined_list[0]

        logger.info("Datasets combined and aligned successfully")
        return df_combined

    def split_and_save_data(self, df_combined):
        logger.info("Splitting data into train and test sets")
        try:
            # Separate features and labels
            X = df_combined[self.common_features]
            y = df_combined['binary_label']

            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Save data
            train_output = self.processed_dir / 'train_processed.csv'
            test_output = self.processed_dir / 'test_processed.csv'

            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            train_df.to_csv(train_output, index=False)
            test_df.to_csv(test_output, index=False)

            logger.info(f"Processed data saved to {self.processed_dir}")

            return train_df, test_df
        except Exception as e:
            logger.error(f"Error splitting and saving data: {e}")
            raise

    def preprocess_all(self):
        try:
            # Get the combined dataset
            df_combined = self.get_combined_dataset()

            # Split and save data
            train_df, test_df = self.split_and_save_data(df_combined)

            logger.info("Data preprocessing completed successfully")
            return train_df, test_df
        except Exception as e:
            logger.error(f"Error in preprocessing all data: {e}")
            raise

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.preprocess_all()

    # Display basic statistics
    print("Training set:")
    print(train_df['binary_label'].value_counts())
    print("Testing set:")
    print(test_df['binary_label'].value_counts())
