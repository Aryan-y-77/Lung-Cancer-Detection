# Lung Cancer Classification Using CNN

## 📌 Project Overview
This project is a **Deep Learning-based image classification model** that detects **lung cancer** using histopathological images. The dataset consists of lung tissue images categorized into different classes. A **Convolutional Neural Network (CNN)** is trained to classify these images into Normal, Adenocarcinoma, and Squamous Cell Carcinoma.

## 📂 Dataset
**Source**: [Kaggle - Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

**Classes:**
- **Lung N** → Normal Lung Tissue
- **Lung A** → Adenocarcinoma (Lung Cancer Type 1)
- **Lung SCC** → Squamous Cell Carcinoma (Lung Cancer Type 2)

## 🛠️ Tech Stack
- **Python**
- **TensorFlow/Keras** for Deep Learning
- **OpenCV** for Image Processing
- **Scikit-learn** for Model Evaluation
- **Matplotlib & Seaborn** for Data Visualization

## 🔧 Setup & Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/lung-cancer-classification.git
   cd lung-cancer-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset using:
   ```python
   import opendatasets as od
   od.download("https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images")
   ```
4. Run the training script:
   ```bash
   python train.py
   ```

## 🏗️ Model Architecture
The CNN model consists of:
- **Convolutional layers** for feature extraction
- **MaxPooling layers** for dimensionality reduction
- **Fully Connected layers** for classification
- **Softmax activation** for multi-class output

## 🎯 Performance Metrics
- **Accuracy:** 95.53% on validation data
- **Loss:** 0.1426
- **Confusion Matrix & Classification Report** used for evaluation

## 🖼️ Testing the Model
You can test the trained model on a custom image:
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('lung_cancer_model.h5')
img = cv2.imread('sample_image.jpeg')
img = cv2.resize(img, (256, 256)) / 255.0
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
print("Predicted Class:", np.argmax(prediction))
```

## 📜 License
This project is open-source and available under the **MIT License**.

## 📬 Contact
For queries, feel free to reach out!
- **Email:** your.email@example.com
- **GitHub:** [your-username](https://github.com/your-username)


