import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

from PIL import Image, ImageTk

from config import EDA_OUTPUT_DIR, REPORT_OUTPUT_DIR


class MacroinvertebrateApp:
    """Desktop GUI for the Macroinvertebrate Image Analysis System."""

    def __init__(self, workflow) -> None:
        # The workflow service does all of the real work (EDA, training, predict).
        self.workflow = workflow

        # Remember which image the user picked.
        self.current_image_path: str | None = None

        # Keep a reference to the displayed image so Python does not
        # garbage-collect it while it is showing.
        self.photo = None

        # Create the main window.
        self.window = tk.Tk()
        self.window.title("Macroinvertebrate Image Analysis System")
        self.window.geometry("800x780")

        # Build all the widgets, then try to load the saved model.
        self._build_widgets()
        self._load_model()

    def _build_widgets(self) -> None:
        """Place all buttons, labels, and frames on the window."""
        # Title at the top of the window.
        title_label = tk.Label(
            self.window,
            text="Macroinvertebrate Image Analysis System",
            font=("Arial", 16, "bold"),
        )
        title_label.pack(pady=10)

        # Group 1: dataset and model actions.
        action_frame = tk.LabelFrame(
            self.window, text="Dataset and Model", padx=10, pady=10
        )
        action_frame.pack(fill="x", padx=10, pady=5)

        tk.Button(
            action_frame,
            text="Show Summary",
            command=self.show_summary,
            width=15,
        ).pack(side="left", padx=5)

        tk.Button(
            action_frame,
            text="Generate EDA",
            command=self.generate_eda,
            width=15,
        ).pack(side="left", padx=5)

        tk.Button(
            action_frame,
            text="Train Model",
            command=self.train_model,
            width=15,
        ).pack(side="left", padx=5)

        tk.Button(
            action_frame,
            text="Open Outputs Folder",
            command=self.open_outputs,
            width=18,
        ).pack(side="left", padx=5)

        # Group 2: image prediction actions.
        predict_frame = tk.LabelFrame(
            self.window, text="Image Prediction", padx=10, pady=10
        )
        predict_frame.pack(fill="x", padx=10, pady=5)

        tk.Button(
            predict_frame,
            text="Open Image",
            command=self.open_image,
            width=15,
        ).pack(side="left", padx=5)

        tk.Button(
            predict_frame,
            text="Predict",
            command=self.predict,
            width=15,
        ).pack(side="left", padx=5)

        tk.Button(
            predict_frame,
            text="View Confusion Matrix",
            command=self.view_confusion_matrix,
            width=22,
        ).pack(side="left", padx=5)

        # Box that shows the chosen image (or a placeholder text).
        self.image_label = tk.Label(
            self.window,
            text="(no image loaded)",
            width=50,
            height=15,
            relief="solid",
            bg="white",
        )
        self.image_label.pack(pady=10)

        # Label that shows the prediction result.
        self.result_label = tk.Label(
            self.window,
            text="Prediction: -",
            font=("Arial", 14),
        )
        self.result_label.pack(pady=5)

        # Scrolling text area used as a small log for messages.
        log_frame = tk.LabelFrame(self.window, text="Log", padx=5, pady=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill="both", expand=True)

        # Status bar at the bottom of the window.
        self.status_label = tk.Label(
            self.window,
            text="Status: starting...",
            anchor="w",
            relief="sunken",
        )
        self.status_label.pack(side="bottom", fill="x")

    # --------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Add a message to the log text area."""
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.window.update_idletasks()

    def _set_status(self, message: str) -> None:
        """Update the status bar at the bottom of the window."""
        self.status_label.config(text=f"Status: {message}")
        self.window.update_idletasks()

    def _load_model(self) -> None:
        """Try to load the trained model when the app starts."""
        try:
            self.workflow.load_model()
            self._set_status("model loaded successfully")
            self._log("Saved model loaded.")
        except FileNotFoundError:
            self._set_status("model not found. Click 'Train Model' first.")
            self._log("No saved model found yet. Train one before predicting.")

    # --------------------------------------------------------------------
    # Button actions: dataset and model
    # --------------------------------------------------------------------

    def show_summary(self) -> None:
        """Show basic dataset statistics in the log area."""
        self._set_status("building dataset summary...")
        try:
            summary = self.workflow.show_summary()
            self._log("\nDataset summary:")
            for key, value in summary.items():
                self._log(f"  {key}: {value}")
            self._set_status("summary ready")
        except Exception as error:
            messagebox.showerror("Summary failed", str(error))
            self._set_status("summary failed")

    def generate_eda(self) -> None:
        """Run all EDA charts and save them to outputs/eda/."""
        self._set_status("generating EDA charts (this may take a moment)...")
        self._log("\nGenerating EDA charts...")
        try:
            summary = self.workflow.generate_eda()
            self._log("EDA done. Files saved in outputs/eda/.")
            self._log(f"Flagged noisy images: {summary.get('noisy_image_count', 0)}")
            self._set_status("EDA generated")
        except Exception as error:
            messagebox.showerror("EDA failed", str(error))
            self._set_status("EDA failed")

    def train_model(self) -> None:
        """Train the baseline classifier on the current dataset."""
        self._set_status("training model (this may take a moment)...")
        self._log("\nTraining baseline classifier...")
        try:
            results = self.workflow.train_model()
            accuracy = results["accuracy"]
            self._log(f"Training accuracy: {accuracy:.4f}")
            self._log("Classification report:")
            self._log(results["report"])
            self._set_status(f"model trained (accuracy {accuracy:.2%})")
        except Exception as error:
            messagebox.showerror("Training failed", str(error))
            self._set_status("training failed")

    def open_outputs(self) -> None:
        """Open the outputs folder in the system file browser."""
        outputs_dir = EDA_OUTPUT_DIR.parent
        if not outputs_dir.exists():
            messagebox.showinfo("Not found", "Outputs folder does not exist yet.")
            return
        # Use the platform-specific way to open a folder.
        if sys.platform == "darwin":
            subprocess.run(["open", str(outputs_dir)])
        elif sys.platform == "win32":
            subprocess.run(["explorer", str(outputs_dir)])
        else:
            subprocess.run(["xdg-open", str(outputs_dir)])
        self._set_status(f"opened {outputs_dir}")

    # --------------------------------------------------------------------
    # Button actions: image prediction
    # --------------------------------------------------------------------

    def open_image(self) -> None:
        """Open a file dialog so the user can pick an image."""
        file_path = filedialog.askopenfilename(
            title="Choose a macroinvertebrate image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )

        # If the user pressed Cancel, do nothing.
        if not file_path:
            return

        self.current_image_path = file_path

        # Open the image with Pillow and resize for display only.
        image = Image.open(file_path)
        image.thumbnail((400, 300))

        # Convert the Pillow image to a Tkinter-friendly format.
        self.photo = ImageTk.PhotoImage(image)

        self.image_label.config(image=self.photo, text="")
        self.result_label.config(text="Prediction: -")
        self._set_status(f"loaded {Path(file_path).name}")
        self._log(f"Loaded image: {file_path}")

    def predict(self) -> None:
        """Predict the class of the loaded image."""
        # Make sure an image has been chosen first.
        if not self.current_image_path:
            messagebox.showwarning(
                "No image",
                "Please open an image before predicting.",
            )
            return

        try:
            prediction = self.workflow.predict_image(self.current_image_path)
            self.result_label.config(text=f"Prediction: {prediction}")
            self._set_status("prediction completed")
            self._log(
                f"Predicted: {prediction} "
                f"(image: {Path(self.current_image_path).name})"
            )
        except FileNotFoundError:
            messagebox.showerror(
                "Model missing",
                "No trained model found. Click 'Train Model' first.",
            )
            self._set_status("model missing")
        except Exception as error:
            messagebox.showerror("Prediction failed", str(error))
            self._set_status("prediction failed")

    def view_confusion_matrix(self) -> None:
        """Open the saved confusion matrix image in a new window."""
        matrix_path = REPORT_OUTPUT_DIR / "confusion_matrix.png"
        if not matrix_path.exists():
            messagebox.showinfo(
                "Not found",
                "Confusion matrix not found. Train the model first.",
            )
            return

        # Create a small extra window that just shows the image.
        viewer = tk.Toplevel(self.window)
        viewer.title("Confusion Matrix")

        image = Image.open(matrix_path)
        image.thumbnail((700, 700))
        photo = ImageTk.PhotoImage(image)

        label = tk.Label(viewer, image=photo)
        # Keep a reference so the image is not garbage-collected.
        label.image = photo
        label.pack(padx=10, pady=10)

        self._set_status("opened confusion matrix")

    # --------------------------------------------------------------------
    # Run loop
    # --------------------------------------------------------------------

    def run(self) -> None:
        """Start the Tkinter event loop (shows the window)."""
        self.window.mainloop()
