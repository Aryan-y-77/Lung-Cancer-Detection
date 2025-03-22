# Lung Cancer Classification Using CNN & ResNet

## 📌 Project Overview
This project involves building **Deep Learning-based image classification models** to detect **lung cancer** using histopathological images. Two approaches are implemented:
1. A **Convolutional Neural Network (CNN)**
2. A **ResNet-based model for improved feature extraction**

Both models classify images into Normal, Adenocarcinoma, and Squamous Cell Carcinoma categories.

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

## 🏗️ Model Architectures
### CNN Model:
- **Convolutional layers** for feature extraction
- **MaxPooling layers** for dimensionality reduction
- **Fully Connected layers** for classification
- **Softmax activation** for multi-class output

### ResNet Model:
- **Pretrained ResNet layers** for robust feature extraction
- **Global Average Pooling** to reduce dimensionality
- **Fully Connected layers** for final classification
- **Softmax activation** for multi-class output

## 🎯 Performance Metrics
### CNN Model:
- **Accuracy:** 95.53% on validation data
- **Loss:** 0.1426

### ResNet Model:
- **Accuracy:** 97.12% on validation data
- **Loss:** 0.0984

Both models were evaluated using the **Confusion Matrix** and **Classification Report**.



## 📈 Training & Validation Performance
The accuracy and loss plots for both models over training epochs:

![Training Accuracy & Loss - CNN](training_plot_cnn.png)
![Training Accuracy & Loss - ResNet](training_plot_resnet.png)

## 🖼️ Testing the Models
You can test a trained model on a custom image:
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


## 💌 Contact
For queries, feel free to reach out!
- **Email:** aryanyadavgr10@example.com
- **GitHub:** [Aryan-y-77](https://github.com/Aryan-y-77)

