# Distracted Driver Detection using CNNs

This project aims to detect and classify distracted driving behavior from in-cabin images using Convolutional Neural Networks (CNNs). We trained a baseline model using Keras and compared its performance against a custom architecture, with real-time deployment plans through a video-to-image classification pipeline.

---

## 📌 Project Objectives

1. Build a baseline CNN model to classify driver behavior into three categories:
   - `safe_driving`
   - `using_phone`
   - `drinking`

2. Address the class imbalance problem through proper augmentation and class weighting.

3. Improve model performance using a custom CNN architecture.

4. Build a video-to-image inference pipeline to deploy the model in real-time driver monitoring scenarios.

---

## 🧠 Dataset Overview

The dataset used is from the [State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data) competition on Kaggle.

- Original data contains images classified into 10 distraction classes (`c0` to `c9`)
- For our study, we filtered it into 3 categories: `safe_driving`, `using_phone`, and `drinking`
- Total filtered images: ~14,000

---

## 🔄 Data Preprocessing

- All images resized to **224x224**
- Augmentation applied only to the **training set**
- Validation and test sets were **untouched** to evaluate generalization
- Split ratio:
  - **Training**: 60%
  - **Validation**: 20%
  - **Testing**: 20%
- CSVs and folder structures were created for each split

---

## 📊 Handling Class Imbalance

Our filtered dataset was imbalanced:

| Class         | Count |
|---------------|-------|
| using_phone   | 9256  |
| safe_driving  | 2489  |
| drinking      | 2325  |

We used `class_weight` in Keras to balance the loss contributions from each class during training.

---

## 🏗️ Baseline Model

A simple CNN was built with:

- 2 convolutional layers
- ReLU activations
- Max pooling
- Flatten → Dense(64) → Dropout(0.5) → Dense(3 with softmax)

### Training Configuration

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: Up to 20 with EarlyStopping
- Augmentations: rotation, zoom, brightness, flips

### Performance (With Class Weights)

| Metric         | Value |
|----------------|-------|
| Test Accuracy  | 96%   |
| F1 (macro avg) | 96%   |
| Precision      | 94% - 99% (per class)
| Recall         | 95% - 98% (per class)

The model achieved high accuracy with balanced performance across all classes after applying class weights.

---

## 🎥 Video-to-Image Pipeline (Planned)

The next phase will involve:

1. Uploading a video and extracting frames
2. Predicting distraction class on each frame using the trained CNN
3. Annotating frames with the prediction
4. Merging frames back into an annotated video
5. Returning both the driver ID and the output video

---

## 📁 Folder Structure

```
distracted_driver_dataset/
│
├── filtered_dataset/ # All filtered images
├── split_data/
│ ├── training/
│ ├── validation/
│ ├── testing/
│ ├── training_data.csv
│ ├── validation_data.csv
│ └── testing_data.csv
├── full_dataset/ # Flattened raw images from original zip
├── image_data.csv # Cleaned metadata
├── notebooks/
│ ├── cnn_baseline_model.ipynb
│ ├── cnn_custom_model.ipynb
│ └── video_inference_pipeline.ipynb
└── README.md
```


---

## ✅ Key Libraries Used

- TensorFlow / Keras
- OpenCV
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn

---

## 🚧 Next Steps

- Build and train a custom CNN architecture
- Compare it with the baseline on performance and inference speed
- Deploy the model in a full pipeline with video input and annotation
- Optionally integrate with a Flask/Streamlit interface for demo purposes

---

## 📌 Authors

- **Nishanth Chennagiri Keerthi**
- **Ashish Mohamed Usman**

---

## 📬 Contact

For collaboration or feedback, feel free to reach out via email or LinkedIn.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/GhwTNp6x)