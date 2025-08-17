# 🧠 Distracted Driver Detection Using CNNs + Flask Web App

This project identifies distracted driving behavior from in-cabin images using deep learning (CNNs) and deploys it via a Flask web interface. It includes:

- A baseline CNN model
- A custom enhanced CNN model
- Real-time video frame classification pipeline
- A working Flask web app with driver ID logging and result dashboard

---

## 📌 Objectives

1. Classify driver behavior into:
   - `safe_driving`
   - `using_phone`
   - `drinking`

2. Handle severe class imbalance using augmentation and `class_weight`.

3. Build and compare both:
   - A baseline CNN
   - A deeper custom CNN with L2 regularization and Batch Normalization

4. Analyze driving behavior from video using frame-by-frame classification.

5. Build a Flask app to allow:
   - Drivers to upload videos
   - Employees to log in and monitor results
   - Dashboard view of uploads and predictions

---

## 🧠 Dataset

We used the [State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection) dataset and filtered it to only include:

- `safe_driving` (from class c0)
- `using_phone` (combined from c1–c4)
- `drinking` (from class c6)

**Filtered Image Count**: ~14,000  
**Split Ratio**:
- Train: 60%
- Validation: 20%
- Test: 20%

---

## 🔄 Preprocessing & Augmentation

- Images resized to 224x224
- Applied augmentations only to training set:
  - Zoom, flip, brightness, rotation
- Used `flow_from_dataframe()` with CSV metadata
- Generated class weights based on distribution:
  
| Class         | Count |
|---------------|-------|
| using_phone   | 9256  |
| safe_driving  | 2489  |
| drinking      | 2325  |

---

## 🏗️ Models Built

### ✅ Baseline CNN (Notebook: `02_baseline_cnn.ipynb`)
- 4 Conv2D blocks
- MaxPooling + Dropout
- Flatten → Dense(128) → Dropout → Dense(3)
- Trained with `categorical_crossentropy` and `Adam`
- Metrics: Accuracy, Precision, Recall

**Performance (w/ Class Weights):**
- Accuracy: ~96%
- Balanced precision/recall/F1 across all classes

---

### 🧪 Custom CNN (Notebook: `03_custom_cnn.ipynb`)
- Deeper ConvNet (6 Conv2D layers)
- BatchNormalization
- L2 Regularization (`kernel_regularizer`)
- Optional DropBlock (experimental)
- Designed for better generalization

**Result:**
- Slight improvement over baseline in validation accuracy and F1
- Much better at avoiding overfitting

---

## 🎥 Video-to-Image Pipeline (Notebook: `04_video_pipeline.ipynb`)
1. Loads trained CNN model
2. Extracts frames from video at fixed FPS
3. Classifies each frame using CNN
4. Tracks consistent offences (≥10 frames of same class)
5. Saves annotated image snapshots and returns report

---

## 🌐 Flask Web Application (`app.py`)

### ✨ Features:
- Video upload form (`index.html`)
- Driver ID submission
- Upload logging (`submissions.xlsx`)
- Employee login system (`login.html`)
- Upload dashboard for employees (`dashboard.html`)
- Analysis route runs the pipeline and renders annotated results (`result.html`)

### 🔧 Key Routes:
| Route             | Description                             |
|------------------|-----------------------------------------|
| `/`              | Home page with upload form              |
| `/submit`        | Handle file + driver ID submission      |
| `/login`         | Employee login                          |
| `/dashboard`     | Admin dashboard showing logs            |
| `/analyze/<vid>` | Execute pipeline and return results     |

### 📁 Output:
- Annotated snapshots saved in `static/combined_snapshots`
- Results displayed in HTML

---

## 📂 Folder Structure

```
distracted_driver_dataset/
│
├── notebooks/
│ ├── 01_data_cleaning.ipynb
│ ├── 02_baseline_cnn.ipynb
│ ├── 03_custom_cnn.ipynb
│ └── 04_video_pipeline.ipynb
│
├── static/
│ ├── uploads/
│ ├── combined_snapshots/
│ ├── style.css
│
├── templates/
│ ├── index.html
│ ├── login.html
│ ├── dashboard.html
│ └── result.html
│
├── filtered_dataset/
├── full_dataset/
├── split_data/
│ ├── training_data.csv / val / test
│
├── raw_data/
│ └── imgs/
│
├── baseline_model.keras
├── best_custom_cnn_model.keras
├── custom_cnn_model.keras
├── image_data.csv
├── app.py
├── requirements.txt
├── readme.md
```

---

## 🛠️ Tech Stack

- Python, TensorFlow/Keras
- Flask (Web backend)
- OpenCV (Video processing)
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn

---

## 📈 Sample Results

Annotated snapshots for detected offences:

- `offence_using_phone_0s_to_5s_combined.jpg`
- `offence_drinking_56s_to_59s_combined.jpg`

Each image combines multiple offending frames with timestamps for clarity.

---

## 💡 Key Learnings

- Managing real-world class imbalance with proper weighting
- Differences in performance between shallow and deep CNNs
- Building a real-time classification pipeline
- Integrating ML with Flask for deployment
- Dynamic notebook execution with `nbclient` for reusability

---

## 🚧 Future Improvements

- Add real-time webcam detection support
- Allow multiple driver IDs in batch
- Integrate frame annotation on actual video (.avi or .mp4)
- Deploy to Heroku or render for demo access

---

## 👥 Authors

- **Nishanth Chennagiri Keerthi**
- **Ashish Mohamed Usman**

---

## 📬 Contact

Want to collaborate or have feedback? 
Contact Details : 

📧 nishanth.keerthi@ucdconnect.ie , nishanthkeerthi@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/nishanth-keerthi)  
🔗 [GitHub](https://github.com/nishanth-keerthi)

📧 ashish.mohamedusman@ucdconnect.ie,ashishusmanmdk@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/ashish-mohamed-usman-5a0a851a5)  
🔗 [GitHub](https://github.com/)


