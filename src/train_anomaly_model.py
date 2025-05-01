import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from data_preprocessing import DataPreprocessor
from feature_extraction import FeatureExtractor
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    def __init__(self, data_dir='/media/weiss/Other/MDIR_USR/Scripts/Disser/system/data', processed_dir='/media/weiss/Other/MDIR_USR/Scripts/Disser/system/data/processed', models_dir='models'):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)

        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination='auto',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        self.preprocessor = DataPreprocessor(data_dir=self.data_dir, processed_dir=self.processed_dir)
        self.feature_extractor = FeatureExtractor()

    def train(self):
        logger.info("Starting anomaly detection model training...")

        # Get preprocessed data
        logger.info("Loading and preprocessing datasets...")
        train_df, test_df = self.preprocessor.preprocess_all()

        # Split features and target
        X_train = train_df.drop('binary_label', axis=1)
        y_train = train_df['binary_label']

        X_test = test_df.drop('binary_label', axis=1)
        y_test = test_df['binary_label']

        # Feature extraction
        logger.info("Extracting features...")
        X_train_features = self.feature_extractor.extract_features(X_train, fit=True)
        X_test_features = self.feature_extractor.extract_features(X_test, fit=False)

        # Train model
        logger.info("Training Isolation Forest...")
        self.model.fit(X_train_features)

        # Get predictions (-1 for anomalies, 1 for normal)
        y_train_pred = self.model.predict(X_train_features)
        y_test_pred = self.model.predict(X_test_features)

        # Convert predictions to binary format (0 for normal, 1 for anomaly)
        y_train_pred = np.where(y_train_pred == 1, 0, 1)
        y_test_pred = np.where(y_test_pred == 1, 0, 1)

        # Calculate anomaly scores
        train_scores = -self.model.score_samples(X_train_features)
        test_scores = -self.model.score_samples(X_test_features)

        # Evaluate results
        logger.info("Evaluating model performance...")

        print("\nTraining Set Results:")
        print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_train, y_train_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_train, y_train_pred))

        print("\nTest Set Results:")
        print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))

        # Analyze anomaly scores
        print("\nAnomaly Score Statistics:")
        print("Training Set:")
        print(pd.Series(train_scores).describe())
        print("\nTest Set:")
        print(pd.Series(test_scores).describe())

        # Save model and artifacts
        logger.info("Saving model and artifacts...")
        joblib.dump(self.model, self.models_dir / 'anomaly_model.pkl')
        joblib.dump(self.feature_extractor.scaler, self.models_dir / 'anomaly_scaler.pkl')

        # Save anomaly scores distribution for threshold tuning
        np.save(self.models_dir / 'train_anomaly_scores.npy', train_scores)
        np.save(self.models_dir / 'test_anomaly_scores.npy', test_scores)

        return self.model, train_scores, test_scores


if __name__ == "__main__":
    detector = AnomalyDetector()
    model, train_scores, test_scores = detector.train()
