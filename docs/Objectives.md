\# Measure It — Project Objectives



\## MO-1: Measurement Accuracy

Build a computer vision system that measures the width and height of an Easypaisa Visa card from an image.



\- Ground-truth width: 85 mm

\- Ground-truth height: 52 mm

\- Target: Mean Absolute Error (MAE) of 5 mm or less on the evaluation instances.

\- Validate measurements against physical measuring-tape readings.



\## MO-2: Reliable Object Detection

Train a segmentation model using my own labelled card images.



\- Collect at least 70 original images.

\- Use a fixed 70/20/10 training, validation and test split.

\- Report IoU, mAP@0.5, mAP@0.5:0.95, precision and recall.

\- Detect when the card or scale reference is missing.

\- Avoid returning unreliable measurements when confidence is too low.



\## MO-3: Reproducibility

Make the project reproducible and easy to run.



\- Document camera calibration and image undistortion.

\- Keep training configuration and random seed.

\- Provide a single command for inference.

\- Document limitations and failure cases.



\*Note: These are project targets, not achieved results. Actual performance will be reported after testing.\*

