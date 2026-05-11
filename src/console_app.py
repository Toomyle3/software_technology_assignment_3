"""Stage 3 entry script: launches the menu-driven console application."""

from services.console_app import ConsoleApp
from services.workflow_service import WorkflowService


def main() -> None:
    """Launch the menu-driven console version of the project."""
    # Build the shared workflow service used by every menu option.
    workflow = WorkflowService()

    # Wrap it in the ConsoleApp and start the menu loop.
    app = ConsoleApp(workflow=workflow)
    app.run()


if __name__ == "__main__":
    main()
