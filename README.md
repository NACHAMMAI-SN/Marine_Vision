# Comparative-Analysis-of-Machine-Learning-Algorithms-for-Marine-Animal-Detection

Marine animal classification is a crucial task in ecological research, biodiversity monitoring, and conservation. This project explores **manual implementations** of machine learning techniques to classify marine animals into five categories: Dolphin, Fish, Lobster, Octopus, and Sea Horse, addressing challenges like underwater distortions and lighting variations.

## Project Overview

This project implements **complete manual implementations** of all machine learning algorithms from scratch without relying on high-level libraries like scikit-learn. Underwater environments present unique challenges for classification tasks due to distortions and inconsistent lighting conditions. This project compares the performance of five **manually implemented** machine learning models:

- **Random Forest (RF)** 
- **Support Vector Machines (SVM)**   
- **K-Means Clustering** 
- **K-Nearest Neighbors (KNN)** 
- **Convolutional Neural Networks (CNN)** 

Through rigorous preprocessing and data augmentation techniques implemented manually, this project demonstrates robust model performance.

### Key Contributions
1. **Complete manual implementations** of all ML algorithms without scikit-learn dependencies
2. Advanced preprocessing techniques including manual PCA, StandardScaler, and data augmentation
3. Manual implementation of neural network layers (Conv2D, Dense, Dropout, GlobalAveragePooling2D)
4. Custom image data generator for batch processing and augmentation
5. Web interface with Flask for real-time predictions
6. Achieved an accuracy of **92%** with manually implemented CNNs


---

## Dataset

The dataset used in this project includes:
- **Training images**: 1241
- **Validation images**: 250  
- **Test images**: 100
- **Categories**: Dolphin, Fish, Lobster, Octopus, Sea Horse

The dataset was preprocessed using **manual implementations** of:
- **StandardScaler**: Manual implementation for feature normalization
- **PCA**: Manual implementation using SVD for dimensionality reduction
- **Image Augmentation**: Manual implementation of rotation, flipping, and shifting
- **Custom Data Generator**: Manual batch processing and image loading

The dataset used in this project is available on Kaggle:
- **Marine Image Dataset for Classification**: [Link to Dataset](https://www.kaggle.com/datasets/ananya12verma/marine-image-dataset-for-classification)

---

## Implementations

### Core Manual Components
1. **ManualStandardScaler**: Custom implementation of feature standardization
2. **ManualPCA**: PCA implementation using Singular Value Decomposition (SVD)
3. **ManualImageDataGenerator**: Custom image loader with augmentation
4. **Manual K-Fold Cross Validation**: For hyperparameter tuning

### Traditional Machine Learning Models (Manual)
1. **Random Forest (RF)**: Manual implementation with decision trees and bootstrap sampling
2. **Support Vector Machines (SVM)**: Manual implementation with RBF and linear kernels
3. **K-Means Clustering**: Manual implementation with centroid optimization
4. **K-Nearest Neighbors (KNN)**: Manual implementation with distance metrics

### Deep Learning Model (Manual)
5. **Convolutional Neural Networks (CNN)**:
   - Manual implementation of Conv2D, Dense, Dropout layers
   - Custom backpropagation and weight updates
   - Manual global average pooling and activation functions

### Manual Preprocessing Techniques
- **Manual PCA**: Dimensionality reduction using SVD
- **Manual Grid Search**: Hyperparameter optimization without scikit-learn
- **Manual Data Augmentation**: Image transformations implemented from scratch

---

## Results

| Model                  | Accuracy (%) | 
|------------------------|--------------|
| Random Forest (RF)     | 75.00        | 
| Support Vector Machines (SVM) | 87.00 | 
| K-Means Clustering     | 65.00        | 
K-Nearest Neighbors (KNN) | 72.00       | 
| Convolutional Neural Networks (CNN) | **92.00**   | 

### Key Insights
- **Manually implemented CNNs** excelled in feature learning, particularly for complex patterns in underwater images
- Traditional manual methods like RF and SVM showed competitive performance
- **Complete manual pipeline** from data loading to prediction demonstrates deep understanding of ML fundamentals
- Custom implementations provide flexibility and transparency in model behavior

---

## Project Structure

```
Comparative-Analysis-of-Machine-Learning-Algorithms-for-Marine-Animal-Detection/
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
├── templates/
│   └── index.html
│
├── uploads/
│   └── (uploaded images will be stored here)
│
├── app.py
├── Final CNN Model.ipynb
├── K_means.ipynb
├── knn.ipynb
├── random_forest.ipynb
├── svm.ipynb
└── README.md
```

---

## Usage

### Requirements
- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- Flask
- PIL (Pillow)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/NACHAMMAI-SN/Marine_Vision.git
   cd Comparative-Analysis-of-Machine-Learning-Algorithms-for-Marine-Animal-Detection
   ```

2. Install dependencies:
   ```bash
   pip install numpy opencv-python matplotlib flask pillow
   ```

3. Run the web application:
   ```bash
   python app.py
   ```

### Manual Implementation Features
- **No scikit-learn dependency**: All algorithms implemented from scratch
- **Transparent code**: Easy to understand and modify each component
- **Educational value**: Demonstrates fundamental ML concepts
- **Production ready**: Web interface for real-time predictions

---

## Conclusion

This project demonstrates the effectiveness of **manually implemented machine learning algorithms** in marine animal classification tasks, achieving state-of-the-art performance with 92% accuracy using custom CNNs. The results underline the potential of deep learning approaches implemented from scratch to address real-world ecological challenges while providing complete transparency and understanding of the underlying mechanisms.

The manual implementations serve as both a practical solution and an educational resource for understanding the inner workings of machine learning algorithms.
