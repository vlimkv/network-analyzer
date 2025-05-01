import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


class ModelTrainer:
    def __init__(self, data_dir='/media/weiss/Other/MDIR_USR/Scripts/Disser/system/data', processed_dir='/media/weiss/Other/MDIR_USR/Scripts/Disser/system/data/processed', models_dir='models'):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            verbose=1  # Show training progress
        )
        self.preprocessor = DataPreprocessor(data_dir=self.data_dir, processed_dir=self.processed_dir)
        self.feature_extractor = FeatureExtractor()

    def train(self):
        logger.info("Starting supervised model training...")

        # Get preprocessed training and testing data
        logger.info("Loading and preprocessing datasets...")
        train_df, test_df = self.preprocessor.preprocess_all()

        # Split features and target variable
        X_train = train_df.drop('binary_label', axis=1)
        y_train = train_df['binary_label']

        X_test = test_df.drop('binary_label', axis=1)
        y_test = test_df['binary_label']

        # Feature extraction
        logger.info("Extracting features...")
        X_train_features = self.feature_extractor.extract_features(X_train, fit=True)
        X_test_features = self.feature_extractor.extract_features(X_test, fit=False)

        # Save feature names and scaler
        feature_columns = X_train_features.columns.tolist()
        joblib.dump(feature_columns, self.models_dir / 'feature_columns.pkl')
        joblib.dump(self.feature_extractor.scaler, self.models_dir / 'scaler.pkl')

        # Train the model
        logger.info("Training the RandomForestClassifier...")
        self.model.fit(X_train_features, y_train)

        # Evaluate on training data
        logger.info("Evaluating on training data...")
        y_train_pred = self.model.predict(X_train_features)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        print("\nTraining Set Results:")
        print(f"Accuracy: {train_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_train, y_train_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_train, y_train_pred))

        # Evaluate on test data
        logger.info("Evaluating on test data...")
        y_test_pred = self.model.predict(X_test_features)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print("\nTest Set Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))

        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))

        # Save the trained model
        model_path = self.models_dir / 'ml_model.pkl'
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

        return self.model


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
