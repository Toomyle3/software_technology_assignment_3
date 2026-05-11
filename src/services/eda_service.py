from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class EDAService:
    """Build EDA charts and summary outputs from the indexed dataframe."""

    def __init__(self, dataframe: pd.DataFrame, output_dir: Path) -> None:
        # The dataframe comes from the DatasetIndexer.
        self.dataframe = dataframe
        # Folder where every chart and CSV will be saved.
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_summary(self) -> dict:
        """Return basic numbers about the dataset."""
        # These four numbers give a quick picture of the dataset.
        return {
            "total_images": int(len(self.dataframe)),
            "total_classes": int(self.dataframe["label"].nunique()),
            "mean_width": float(self.dataframe["width"].mean()),
            "mean_height": float(self.dataframe["height"].mean()),
        }

    def save_class_distribution(self) -> Path:
        """Plot how many images each class has (class imbalance check)."""
        # Count the number of images for each class label.
        counts = self.dataframe["label"].value_counts()

        # Draw a horizontal bar chart of class counts.
        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts.values, y=counts.index, color="steelblue")
        plt.title("Class Imbalance (Image Count per Class)")
        plt.xlabel("Number of Images")
        plt.ylabel("Class")
        plt.tight_layout()

        # Save the figure and a CSV with the exact numbers.
        output_path = self.output_dir / "class_distribution.png"
        plt.savefig(output_path)
        plt.close()
        counts.to_csv(self.output_dir / "class_distribution.csv")
        return output_path

    def save_image_size_distribution(self) -> Path:
        """Plot image width/height and aspect ratio per class."""
        # Add aspect ratio so we can spot odd-shaped images.
        df = self.dataframe.copy()
        df["aspect_ratio"] = df["width"] / df["height"]

        # Save a table of mean values per class.
        stats = df.groupby("label")[["width", "height", "aspect_ratio"]].mean()
        stats.to_csv(self.output_dir / "dimension_stats.csv")

        # Two histograms side by side: width and height.
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df["width"], bins=20, ax=axes[0])
        sns.histplot(df["height"], bins=20, ax=axes[1])
        axes[0].set_title("Image Width Distribution")
        axes[1].set_title("Image Height Distribution")
        plt.tight_layout()

        output_path = self.output_dir / "image_size_distribution.png"
        plt.savefig(output_path)
        plt.close()

        # A scatter plot makes per-class size differences easy to see.
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="width", y="height", hue="label", alpha=0.7)
        plt.title("Image Dimensions by Class")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "dimensions_scatter.png")
        plt.close()

        # A box plot of aspect ratio helps spot odd shapes per class.
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="label", y="aspect_ratio")
        plt.title("Aspect Ratio per Class")
        plt.xlabel("Class")
        plt.ylabel("Aspect Ratio (width / height)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "aspect_ratio_box.png")
        plt.close()

        return output_path

    def save_pixel_intensity_distribution(self) -> Path:
        """Plot how bright/dark images are (helps decide grayscale use)."""
        # Mean pixel intensity per image, from 0 (black) to 255 (white).
        rows = []
        for _, row in self.dataframe.iterrows():
            image = cv2.imread(row["file_path"], cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            rows.append(
                {"label": row["label"], "mean_intensity": float(image.mean())}
            )

        intensity_df = pd.DataFrame(rows)
        intensity_df.to_csv(self.output_dir / "pixel_intensity.csv", index=False)

        # Histogram across all images.
        plt.figure(figsize=(10, 6))
        sns.histplot(data=intensity_df, x="mean_intensity", bins=30, color="purple")
        plt.title("Pixel Intensity Distribution (All Images)")
        plt.xlabel("Mean Pixel Intensity")
        plt.ylabel("Number of Images")
        plt.tight_layout()
        output_path = self.output_dir / "pixel_intensity_hist.png"
        plt.savefig(output_path)
        plt.close()

        # Box plot per class to compare brightness between classes.
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=intensity_df, x="label", y="mean_intensity")
        plt.title("Mean Pixel Intensity per Class")
        plt.xlabel("Class")
        plt.ylabel("Mean Pixel Intensity")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "pixel_intensity_box.png")
        plt.close()

        return output_path

    def save_sample_grid(self, samples_per_class: int = 3) -> Path:
        """Save a grid of example images, one row per class."""
        labels = sorted(self.dataframe["label"].unique())
        num_classes = len(labels)

        # squeeze=False keeps axes as a 2D array even when sizes are 1.
        fig, axes = plt.subplots(
            num_classes,
            samples_per_class,
            figsize=(samples_per_class * 3, num_classes * 3),
            squeeze=False,
        )

        # Hide every axis first so empty cells don't show plot frames.
        for ax in axes.flat:
            ax.axis("off")

        # Fill in a few sample images for each class.
        for row_index, label in enumerate(labels):
            class_rows = self.dataframe[self.dataframe["label"] == label].head(
                samples_per_class
            )
            for col_index, (_, row) in enumerate(class_rows.iterrows()):
                image = cv2.imread(row["file_path"])
                if image is None:
                    continue
                # OpenCV reads as BGR, but matplotlib expects RGB.
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                axes[row_index, col_index].imshow(image_rgb)
                axes[row_index, col_index].set_title(label, fontsize=9)

        plt.tight_layout()
        output_path = self.output_dir / "sample_images.png"
        plt.savefig(output_path)
        plt.close()
        return output_path

    def save_noisy_images(
        self,
        dark_threshold: float = 30.0,
        bright_threshold: float = 230.0,
    ) -> pd.DataFrame:
        """Flag images that are very dark, very bright, or unusually sized."""
        flagged = []

        # Use the median size as a typical reference.
        median_width = self.dataframe["width"].median()
        median_height = self.dataframe["height"].median()

        for _, row in self.dataframe.iterrows():
            image = cv2.imread(row["file_path"], cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            mean_intensity = float(image.mean())
            reasons = []

            if mean_intensity < dark_threshold:
                reasons.append("too_dark")
            if mean_intensity > bright_threshold:
                reasons.append("too_bright")

            # Flag images that are much smaller or larger than typical.
            if row["width"] < median_width * 0.5 or row["width"] > median_width * 2:
                reasons.append("unusual_width")
            if row["height"] < median_height * 0.5 or row["height"] > median_height * 2:
                reasons.append("unusual_height")

            if reasons:
                flagged.append(
                    {
                        "file_path": row["file_path"],
                        "label": row["label"],
                        "mean_intensity": mean_intensity,
                        "width": row["width"],
                        "height": row["height"],
                        "reasons": ", ".join(reasons),
                    }
                )

        noisy_df = pd.DataFrame(flagged)
        noisy_df.to_csv(self.output_dir / "noisy_images.csv", index=False)
        return noisy_df

    def generate_all(self) -> dict:
        """Run every EDA step and return a summary."""
        # Run all charts and save all files at once.
        self.save_class_distribution()
        self.save_image_size_distribution()
        self.save_pixel_intensity_distribution()
        self.save_sample_grid()
        noisy_df = self.save_noisy_images()

        summary = self.build_summary()
        summary["noisy_image_count"] = int(len(noisy_df))
        return summary
