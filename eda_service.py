import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class EDAService:
    def __init__(self, df, output_dir):
        self.df = df
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def show_basic_info(self):
        print("\nFirst 5 rows:")
        print(self.df.head())

        print("\nLast 5 rows:")
        print(self.df.tail())

        print("\nDataset information:")
        print(self.df.info())

        print("\nClass distribution:")
        print(self.df["label"].value_counts())

    def save_class_distribution_chart(self):
        class_counts = self.df["label"].value_counts()

        plt.figure(figsize=(10, 6))
        class_counts.plot(kind="bar")
        plt.title("Class Distribution")
        plt.xlabel("Class Label")
        plt.ylabel("Number of Images")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(self.output_dir / "class_distribution.png")
        plt.close()

    def save_image_size_distribution_chart(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df["width"], self.df["height"])
        plt.title("Image Size Distribution")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.tight_layout()

        plt.savefig(self.output_dir / "image_size_distribution.png")
        plt.close()

    def save_summary_statistics(self):
        summary = {
            "total_images": [len(self.df)],
            "total_classes": [self.df["label"].nunique()],
            "classes": [", ".join(sorted(self.df["label"].unique()))]
        }

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(self.output_dir / "dataset_summary.csv", index=False)

    def split_dataset(self):
        train_df, temp_df = train_test_split(
            self.df,
            train_size=0.8,
            shuffle=True,
            random_state=123,
            stratify=self.df["label"]
        )

        valid_df, test_df = train_test_split(
            temp_df,
            train_size=0.6,
            shuffle=True,
            random_state=123,
            stratify=temp_df["label"]
        )

        print("\nTraining images:", len(train_df))
        print("Validation images:", len(valid_df))
        print("Testing images:", len(test_df))

        return train_df, valid_df, test_df

    def create_generators(self, train_df, valid_df, test_df, img_size=(224, 224), batch_size=16):
        image_generator = ImageDataGenerator(rescale=1.0 / 255)

        train_gen = image_generator.flow_from_dataframe(
            train_df,
            x_col="file_path",
            y_col="label",
            target_size=img_size,
            color_mode="rgb",
            class_mode="categorical",
            shuffle=True,
            batch_size=batch_size
        )

        valid_gen = image_generator.flow_from_dataframe(
            valid_df,
            x_col="file_path",
            y_col="label",
            target_size=img_size,
            color_mode="rgb",
            class_mode="categorical",
            shuffle=True,
            batch_size=batch_size
        )

        test_gen = image_generator.flow_from_dataframe(
            test_df,
            x_col="file_path",
            y_col="label",
            target_size=img_size,
            color_mode="rgb",
            class_mode="categorical",
            shuffle=False,
            batch_size=batch_size
        )

        return train_gen, valid_gen, test_gen

    def save_sample_grid(self, train_gen):
        images, labels = next(train_gen)
        class_names = list(train_gen.class_indices.keys())

        plt.figure(figsize=(12, 12))

        for i in range(min(16, len(images))):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i])

            label_index = labels[i].argmax()
            class_name = class_names[label_index]

            plt.title(class_name)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_grid.png")
        plt.close()
        