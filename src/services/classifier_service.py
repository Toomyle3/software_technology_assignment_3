"""
*******************************
Author:
u3332252 Assessment_3_group_4 12/05/2024
Question:
Stage 2: Classification service: trains, evaluates, saves, and reuses
the Random Forest model that predicts the macroinvertebrate class.
*******************************
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from services.image_preprocessor import ImagePreprocessor


class ClassifierService:
    """Train, evaluate, save, and reuse the macroinvertebrate classifier."""

    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        model_output_dir: Path,
        report_output_dir: Path,
    ) -> None:
        self.preprocessor = preprocessor
        self.model_output_dir = model_output_dir
        self.report_output_dir = report_output_dir

        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.report_output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_features(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert image paths and labels into features and target labels."""
        features = []
        labels = []

        for _, row in dataframe.iterrows():
            try:
                image_features = self.preprocessor.transform(row["file_path"])
                features.append(image_features)
                labels.append(str(row["label"]).strip())
            except ValueError as error:
                print(f"Skipped image: {error}")

        return np.array(features), np.array(labels)

    def train(self, dataframe: pd.DataFrame) -> dict[str, object]:
        """Train classifier and save all evaluation outputs."""
        X, y = self.prepare_features(dataframe)

        if len(X) == 0:
            raise ValueError("No valid images found for training.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions)
        labels = sorted(np.unique(y))

        self.save_model()
        self.save_classification_report(report, accuracy)
        self.save_confusion_matrix(matrix, labels)

        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": matrix,
            "labels": labels,
        }

    def save_model(self, file_name: str = "macro_classifier.joblib") -> Path:
        """Save trained model using joblib."""
        model_path = self.model_output_dir / file_name
        joblib.dump(self.model, model_path)
        return model_path

    def load_model(self, file_name: str = "macro_classifier.joblib") -> None:
        """Load saved model."""
        model_path = self.model_output_dir / file_name

        if not model_path.exists():
            raise FileNotFoundError("Model not found. Train the model first.")

        self.model = joblib.load(model_path)

    def save_classification_report(
        self,
        report: str,
        accuracy: float,
    ) -> Path:
        """Save classification report as text file."""
        report_path = self.report_output_dir / "classification_report.txt"

        content = (
            "Macroinvertebrate Classification Report\n"
            "======================================\n\n"
            f"Model Accuracy: {accuracy:.4f}\n\n"
            f"{report}"
        )

        report_path.write_text(content, encoding="utf-8")
        return report_path

    def save_confusion_matrix(
        self,
        matrix: np.ndarray,
        labels: list[str],
    ) -> Path:
        """Save confusion matrix heatmap."""
        output_path = self.report_output_dir / "confusion_matrix.png"

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            xticklabels=labels,
            yticklabels=labels,
        )

        plt.title("Macroinvertebrate Classification Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return output_path

    def predict_image(self, file_path: str) -> str:
        """Predict class for one image."""
        image_features = self.preprocessor.transform(file_path).reshape(1, -1)
        prediction = self.model.predict(image_features)[0]

        return str(prediction)

    def save_batch_confusion_matrix(
        self,
        y_true: list[str],
        y_pred: list[str],
        file_name: str = "batch_confusion_matrix.png",
    ) -> Path:
        """Save a confusion matrix image for an ad-hoc batch of predictions."""
        labels = sorted(set(y_true) | set(y_pred))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)

        output_path = self.report_output_dir / file_name

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            xticklabels=labels,
            yticklabels=labels,
        )

        plt.title("Batch Prediction Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return output_path