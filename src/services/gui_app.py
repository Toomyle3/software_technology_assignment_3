import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk


class MacroinvertebrateApp:
    """Simple Tkinter desktop app: open an image and predict its class."""

    def __init__(self, workflow) -> None:
        # The workflow service handles model loading and prediction.
        self.workflow = workflow

        # Remember which image the user picked.
        self.current_image_path: str | None = None

        # Keep a reference to the PhotoImage so Python does not delete it
        # (Tkinter needs the image object to stay alive while shown).
        self.photo = None

        # Create the main window.
        self.window = tk.Tk()
        self.window.title("Macroinvertebrate Classifier")
        self.window.geometry("600x650")

        # Build the widgets and try to load the saved model.
        self._build_widgets()
        self._load_model()

    def _build_widgets(self) -> None:
        """Place all the buttons, labels, and frames on the window."""
        # Title at the top of the window.
        title_label = tk.Label(
            self.window,
            text="Macroinvertebrate Classifier",
            font=("Arial", 16, "bold"),
        )
        title_label.pack(pady=10)

        # A row of buttons.
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=5)

        open_button = tk.Button(
            button_frame,
            text="Open Image",
            command=self.open_image,
            width=15,
        )
        open_button.pack(side="left", padx=5)

        predict_button = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict,
            width=15,
        )
        predict_button.pack(side="left", padx=5)

        # Box that shows the chosen image (or a placeholder text).
        self.image_label = tk.Label(
            self.window,
            text="(no image loaded)",
            width=50,
            height=20,
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
        self.result_label.pack(pady=10)

        # Status bar at the bottom of the window.
        self.status_label = tk.Label(
            self.window,
            text="Status: starting...",
            anchor="w",
            relief="sunken",
        )
        self.status_label.pack(side="bottom", fill="x")

    def _load_model(self) -> None:
        """Try to load the trained model when the app starts."""
        try:
            self.workflow.load_model()
            self.status_label.config(text="Status: model loaded successfully")
        except FileNotFoundError:
            # If no model exists yet, tell the user to train first.
            self.status_label.config(
                text="Status: model not found. Run main.py to train first."
            )

    def open_image(self) -> None:
        """Open a file dialog so the user can pick an image."""
        # Ask the user to choose an image file.
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

        # Open the image with Pillow and resize it for display only.
        image = Image.open(file_path)
        image.thumbnail((400, 400))

        # Convert the Pillow image to a Tkinter-friendly format.
        self.photo = ImageTk.PhotoImage(image)

        # Show the image in the label.
        self.image_label.config(image=self.photo, text="")

        # Reset the prediction text and update the status bar.
        self.result_label.config(text="Prediction: -")
        self.status_label.config(text=f"Status: loaded {Path(file_path).name}")

    def predict(self) -> None:
        """Use the workflow service to predict the class of the loaded image."""
        # Make sure an image has been chosen first.
        if not self.current_image_path:
            messagebox.showwarning(
                "No image",
                "Please open an image before predicting.",
            )
            return

        try:
            # Ask the workflow service for a prediction.
            prediction = self.workflow.predict_image(self.current_image_path)
            self.result_label.config(text=f"Prediction: {prediction}")
            self.status_label.config(text="Status: prediction completed")
        except FileNotFoundError:
            messagebox.showerror(
                "Model missing",
                "No trained model found. Please run main.py to train the model.",
            )
        except Exception as error:
            messagebox.showerror("Prediction failed", str(error))

    def run(self) -> None:
        """Start the Tkinter event loop (shows the window)."""
        self.window.mainloop()
