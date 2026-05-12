"""
*******************************
Author:
u3290048 Assessment_3_group_4 12/05/2024
Question:
Stage 3: menu-driven console application. Same actions as the GUI but
runs entirely in the terminal — useful when a display is not available.
*******************************
"""

from services.workflow_service import WorkflowService


class ConsoleApp:
    """Simple menu-driven console version of the project."""

    def __init__(self, workflow: WorkflowService) -> None:
        # The workflow service knows how to do every action.
        self.workflow = workflow

    def run(self) -> None:
        """Show the menu in a loop until the user chooses to exit."""
        while True:
            # Print the menu options every time.
            print("\nMacroinvertebrate Image Analysis System")
            print("---------------------------------------")
            print("1. Show dataset summary")
            print("2. Generate EDA outputs (charts and CSVs)")
            print("3. Train baseline classifier")
            print("4. Predict an image")
            print("5. Exit")

            # Read the user's choice.
            choice = input("Select an option: ").strip()

            if choice == "1":
                self._handle_summary()
            elif choice == "2":
                self._handle_eda()
            elif choice == "3":
                self._handle_train()
            elif choice == "4":
                self._handle_predict()
            elif choice == "5":
                print("Goodbye.")
                break
            else:
                print("Invalid option. Please try again.")

    def _handle_summary(self) -> None:
        """Option 1: print dataset summary."""
        try:
            self.workflow.show_summary()
        except Exception as error:
            print(f"Error: {error}")

    def _handle_eda(self) -> None:
        """Option 2: run the EDA charts."""
        try:
            self.workflow.generate_eda()
        except Exception as error:
            print(f"Error: {error}")

    def _handle_train(self) -> None:
        """Option 3: train the baseline model."""
        try:
            self.workflow.train_model()
        except Exception as error:
            print(f"Error: {error}")

    def _handle_predict(self) -> None:
        """Option 4: predict the class of one image."""
        # Ask the user for a path to an image file.
        image_path = input("Enter image path: ").strip()
        if not image_path:
            print("No path entered.")
            return

        try:
            prediction = self.workflow.predict_image(image_path)
            print(f"Predicted class: {prediction}")
        except FileNotFoundError:
            print("No trained model found. Please train the model first (option 3).")
        except ValueError as error:
            print(f"Could not read the image: {error}")
        except Exception as error:
            print(f"Error: {error}")
