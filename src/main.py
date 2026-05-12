"""
*******************************
Author:
u3332252 Assessment_3_group_4 12/05/2024
u3214187 Assessment_3_group_4 12/05/2024
Question:
Stage 1 + 2: Batch entry script: runs Stage 1 EDA + Stage 2 training end-to-end
without any user interface.
*******************************
"""

from services.workflow_service import WorkflowService


def main() -> None:
    """Run the default Stage 1 + Stage 2 workflow without a GUI."""
    print("Macroinvertebrate Image Analysis System")
    print("=======================================")

    # The workflow service does everything: load data, EDA, training.
    workflow = WorkflowService()
    workflow.run_full_pipeline()

    print("\nAll done.")
    print("EDA outputs:    outputs/eda/")
    print("Model files:    outputs/models/")
    print("Training reports: outputs/reports/")


if __name__ == "__main__":
    main()
