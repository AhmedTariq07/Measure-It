\# Measure It — Definition of Done



\## 1. Project Documentation

\- \[ ] Plan.md completed and committed.

\- \[ ] Objectives.md completed.

\- \[ ] Definition-of-Done.md completed.

\- \[ ] README.md includes setup and usage instructions.



\## 2. Camera Calibration

\- \[ ] Capture at least 20 checkerboard calibration images.

\- \[ ] Calculate camera matrix and distortion coefficients.

\- \[ ] Save calibration results.

\- \[ ] Report reprojection error.



\## 3. Dataset

\- \[ ] Collect at least 70 original card images.

\- \[ ] Label images with segmentation masks.

\- \[ ] Create fixed 70/20/10 dataset split.

\- \[ ] Keep test images separate from training and tuning.



\## 4. Model Training

\- \[ ] Train an allowed segmentation model using my own images.

\- \[ ] Save model weights and training configuration.

\- \[ ] Report IoU, mAP@0.5, mAP@0.5:0.95, precision and recall.



\## 5. Measurement Pipeline

\- \[ ] Undistort input image.

\- \[ ] Segment the card.

\- \[ ] Detect the separate scale reference.

\- \[ ] Convert pixel measurements into millimetres.

\- \[ ] Generate measurement overlay.

\- \[ ] Output width, height and confidence.

\- \[ ] Handle missing card or reference.



\## 6. Testing and Evaluation

\- \[ ] Compare results against physical measurements on at least 10 instances.

\- \[ ] Calculate MAE and mean percentage error.

\- \[ ] Compare raw and undistorted image measurements.

\- \[ ] Test at least one failure condition.

\- \[ ] Evaluate confidence threshold and unreliable measurement rejection.



\## 7. Submission

\- \[ ] Push project to public GitHub repository.

\- \[ ] Create pull request from dev-test01 to main.

\- \[ ] Record Loom demonstration of no more than 5 minutes.

\- \[ ] Submit pull request and Loom link by email.



\*Items will be checked only when completed and supported by actual evidence. No results will be fabricated.\*

