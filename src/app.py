"""Stage 3 entry script: launches the Tkinter desktop application."""

from services.gui_app import MacroinvertebrateApp
from services.workflow_service import WorkflowService


def main() -> None:
    """Launch the Tkinter desktop app (Stage 3 deployment)."""
    # Build the shared workflow service that handles model and predictions.
    workflow = WorkflowService()

    # Pass the workflow into the GUI app and start the window.
    app = MacroinvertebrateApp(workflow=workflow)
    app.run()


if __name__ == "__main__":
    main()
