# Manual Testing Evidence

This document records the manual tests performed on the
Macroinvertebrate Image Analysis System. We did not use a formal testing
framework. Instead we ran each scenario by hand and noted the outcome.

## Test Scenarios

| # | Scenario | Input / Action | Expected Result | Actual Result | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | Run full pipeline | `python main.py` with valid dataset | Summary printed, EDA files saved, model trained, accuracy reported | Same as expected | Passed |
| 2 | Missing dataset folder | Rename `data/raw/macro_small` and run `python main.py` | Friendly message "No images found" | Same as expected | Passed |
| 3 | Predict before training | Run `python app.py` with no saved model | Status bar shows "model not found. Run main.py to train first." | Same as expected | Passed |
| 4 | GUI: open and predict | Run `python app.py`, click "Open Image", choose a class image, click "Predict" | Image is shown, predicted class is displayed | Same as expected | Passed |
| 5 | GUI: predict with no image | Run `python app.py`, click "Predict" without opening an image | Warning popup "Please open an image before predicting." | Same as expected | Passed |
| 6 | GUI: invalid image file | Open a non-image file (e.g. a `.txt`) | Error popup with a clear message | Same as expected | Passed |
| 7 | Console: invalid menu choice | Run `python console_app.py`, enter `9` | Message "Invalid option. Please try again." | Same as expected | Passed |
| 8 | Console: predict with bad path | Choose option 4, enter a non-existent path | Message "Could not read the image: ..." | Same as expected | Passed |
| 9 | Console: predict before training | Choose option 4 before option 3 | Message "No trained model found. Please train the model first (option 3)." | Same as expected | Passed |
| 10 | Repeat predictions | In the GUI, open several different images and click Predict each time | Each prediction updates without restart | Same as expected | Passed |

## Notes

- Stage 1 EDA outputs are saved to `outputs/eda/` and were checked
  visually after each test run.
- Stage 2 evaluation outputs are saved to `outputs/reports/`.
- The trained model is saved to `outputs/models/macro_classifier.joblib`
  and is reused by both `app.py` and `console_app.py`.
