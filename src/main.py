from services.classifier_service import ClassifierService
from config import MODEL_OUTPUT_DIR, REPORT_OUTPUT_DIR
from services.dataset_indexer import DatasetIndexer
from services.image_preprocessor import ImagePreprocessor


def main() -> None:
    """Run Stage 2 training and evaluation."""
    print("Starting Stage 2: Predictive Analytics")

    indexer = DatasetIndexer()
    dataframe = indexer.build_dataframe()

    print(f"Total images found: {len(dataframe)}")

    if dataframe.empty:
        print("No images found. Check your dataset folder path.")
        return

    print(f"Classes found: {dataframe['label'].nunique()}")
    print(dataframe["label"].value_counts())

    preprocessor = ImagePreprocessor()

    classifier = ClassifierService(
        preprocessor=preprocessor,
        model_output_dir=MODEL_OUTPUT_DIR,
        report_output_dir=REPORT_OUTPUT_DIR,
    )

    results = classifier.train(dataframe)

    print("\nTraining completed successfully.")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results["report"])

    print("\nSaved outputs:")
    print("outputs/models/macro_classifier.joblib")
    print("outputs/reports/classification_report.txt")
    print("outputs/reports/confusion_matrix.png")


if __name__ == "__main__":
    main()
    