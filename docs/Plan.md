\# Measure It — Project Plan



\## 1. Objective

Build a computer vision system that measures the width and height of an Easypaisa Visa card in millimetres from a photograph.



\*\*Target object:\*\* Easypaisa Visa card  

\*\*Ground-truth width:\*\* 85 mm  

\*\*Ground-truth height:\*\* 52 mm



\## 2. Approach Chosen

\- Use OpenCV to calibrate the camera using at least 20 checkerboard images.

\- Undistort photographs before measurement.

\- Capture at least 70 original photographs of the card.

\- Label the card using segmentation masks.

\- Train a segmentation model using PyTorch and a pretrained backbone. The model will be fine-tuned on my own dataset.

\- Use a measuring tape as a separate scale reference, positioned beside the card in the same plane.

\- Convert pixel measurements into millimetres using the detected reference scale.

\- Evaluate measurements against physical ruler measurements.



\## 3. Dataset and Evaluation

\- Collect at least 70 original images.

\- Use a fixed 70% training, 20% validation and 10% testing split.

\- Save the split file lists before training.

\- Keep test images separate from training and model selection.

\- Evaluate IoU, mAP@0.5, mAP@0.5:0.95, precision and recall.



\## 4. Alternatives Rejected

\- Ultralytics YOLO and Roboflow-trained models are excluded because the assessment prohibits them.

\- Manual measurement alone is rejected because the objective is automated measurement.

\- Assuming nominal card dimensions without physical measurement is rejected.



\## 5. Cost and Constraints

\- Use my existing laptop and phone camera.

\- Use free and open-source Python libraries and tools.

\- Use my measuring tape as the physical reference.

\- No paid annotation or model-training service is planned.



\## 6. Known Risks

\- Camera angle, lighting, blur and card position may affect accuracy.

\- Measuring-tape markings may be difficult to detect automatically.

\- A small dataset may limit model performance.



\## 7. Success Criteria

\- Reproducible camera calibration and image undistortion.

\- A trained segmentation model using my own images.

\- Automated width and height measurement in millimetres.

\- Comparison against physical measurements on at least 10 test instances.

\- Honest reporting of errors, limitations and failure cases.

