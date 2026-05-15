"""
*******************************
Author:
u3290048 Assessment_3_group_4 12/05/2024
Question:
Stage 3: Implementation and Deployment
*******************************
"""


from pathlib import Path

import pandas as pd

from config import EDA_OUTPUT_DIR, MODEL_OUTPUT_DIR, RAW_DATA_DIR, REPORT_OUTPUT_DIR
from services.classifier_service import ClassifierService
from services.dataset_indexer import DatasetIndexer
from services.eda_service import EDAService
from services.image_preprocessor import ImagePreprocessor


class WorkflowService:
    """Coordinate the shared steps used by the batch, GUI, and console apps."""

    def __init__(self) -> None:
        # Make sure all output folders exist before we start.
        EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Build the helper services that do the real work.
        self.indexer = DatasetIndexer()
        self.preprocessor = ImagePreprocessor()
        self.classifier = ClassifierService(
            preprocessor=self.preprocessor,
            model_output_dir=MODEL_OUTPUT_DIR,
            report_output_dir=REPORT_OUTPUT_DIR,
        )

        # We cache the dataframe so we do not re-scan the dataset every time.
        self.dataframe: pd.DataFrame | None = None
        # Track whether the trained model has been loaded into memory.
        self.model_loaded = False

    def load_dataframe(self) -> pd.DataFrame:
        """Scan the dataset folder once and reuse the result later."""
        if self.dataframe is None:
            self.dataframe = self.indexer.build_dataframe()
        return self.dataframe

    def show_summary(self) -> dict:
        """Print and return basic dataset summary numbers."""
        dataframe = self.load_dataframe()
        eda = EDAService(dataframe, EDA_OUTPUT_DIR)
        summary = eda.build_summary()
        print("\nDataset summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return summary

    def generate_eda(self) -> dict:
        """Run all EDA steps and save charts to outputs/eda/."""
        dataframe = self.load_dataframe()
        eda = EDAService(dataframe, EDA_OUTPUT_DIR)
        summary = eda.generate_all()
        print(f"\nEDA finished. Files saved in {EDA_OUTPUT_DIR}")
        return summary

    def train_model(self) -> dict:
        """Train the classifier on the current dataset."""
        dataframe = self.load_dataframe()
        results = self.classifier.train(dataframe)
        self.model_loaded = True
        print(f"\nTraining accuracy: {results['accuracy']:.4f}")
        print(results["report"])
        return results

    def load_model(self) -> None:
        """Load the trained model from disk if it has not been loaded yet."""
        if self.model_loaded:
            return
        self.classifier.load_model()
        self.model_loaded = True

    def predict_image(self, file_path: str) -> str:
        """Predict the class for a single image using the trained model."""
        # Make sure the model is loaded before predicting.
        self.load_model()
        return self.classifier.predict_image(file_path)

    def predict_images(self, file_paths: list[str]) -> dict:
        """Predict classes for multiple images.

        Returns a dict with:
          - results: list of (path, true_label_or_None, prediction_or_error)
          - accuracy: float in [0,1] over images with inferable truth, or None
          - batch_cm_path: Path to saved batch confusion matrix, or None
        """
        self.load_model()

        # Known labels from the dataset folder layout (each sub-folder of
        # RAW_DATA_DIR is one class label). Used to decide if an image's
        # parent folder name is a real class label we can score against.
        known_labels: set[str] = set()
        if RAW_DATA_DIR.exists():
            known_labels = {p.name for p in RAW_DATA_DIR.iterdir() if p.is_dir()}

        results: list[tuple[str, str | None, str]] = []
        y_true: list[str] = []
        y_pred: list[str] = []

        for file_path in file_paths:
            parent_name = Path(file_path).parent.name
            true_label = parent_name if parent_name in known_labels else None

            try:
                prediction = self.classifier.predict_image(file_path)
            except Exception as error:
                results.append((file_path, true_label, f"ERROR: {error}"))
                continue

            results.append((file_path, true_label, prediction))
            if true_label is not None:
                y_true.append(true_label)
                y_pred.append(prediction)

        accuracy = None
        batch_cm_path = None
        if y_true:
            correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            accuracy = correct / len(y_true)
            batch_cm_path = self.classifier.save_batch_confusion_matrix(
                y_true, y_pred
            )

        return {
            "results": results,
            "accuracy": accuracy,
            "batch_cm_path": batch_cm_path,
        }

    def run_full_pipeline(self) -> None:
        """Run the default Stage 1 + Stage 2 workflow in one call."""
        self.show_summary()
        self.generate_eda()
        self.train_model()
