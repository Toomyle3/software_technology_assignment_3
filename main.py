from config import DATASET_DIR, EDA_OUTPUT_DIR, IMG_SIZE, BATCH_SIZE
from dataset_indexer import DatasetIndexer
from eda_service import EDAService


def main():
    print("Starting Dataset Indexing and EDA...")

    indexer = DatasetIndexer(DATASET_DIR)
    df = indexer.build_dataframe()

    if df.empty:
        print("No images found. Please check your dataset folder path.")
        return

    eda = EDAService(df, EDA_OUTPUT_DIR)

    # EDA Q1 and Q2
    eda.show_basic_info()

    # EDA charts and summary
    eda.save_class_distribution_chart()
    eda.save_image_size_distribution_chart()
    eda.save_summary_statistics()

    # EDA Q3
    train_df, valid_df, test_df = eda.split_dataset()

    # EDA Q4 and Q5
    train_gen, valid_gen, test_gen = eda.create_generators(
        train_df,
        valid_df,
        test_df,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    eda.save_sample_grid(train_gen)

    print("\nEDA completed successfully.")
    print("Outputs saved in:", EDA_OUTPUT_DIR)


if __name__ == "__main__":
    main()