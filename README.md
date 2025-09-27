# 🧠 Brain Tumor MRI Classifier

This project uses a Convolutional Neural Network (CNN) to classify brain tumors from MRI images. It leverages TensorFlow and Keras to build, train, and evaluate a deep learning model capable of distinguishing between different tumor types.

## 📌 Project Overview

This project demonstrates how to:

- Load and preprocess MRI images using TensorFlow's `image_dataset_from_directory`
- Build a CNN model for multi-class classification
- Train and validate the model using a split dataset
- Visualize training performance
- Evaluate model accuracy on unseen data

## 📂 Dataset

The dataset should be organized in the following structure:

Brain Tumor/ 
└── Training/ 
├── TumorType1/ 
├── TumorType2/ 
└── TumorType3/

Each subfolder under `Training` should contain MRI images corresponding to a specific tumor class.

## 🧪 Model Architecture

The CNN model includes:

- `Rescaling`: Normalize pixel values to [0, 1]
- `Conv2D` layers: Extract spatial features
- `MaxPooling`: Downsample feature maps
- `Flatten`: Convert 2D features to 1D
- `Dense` layers: Learn complex patterns
- `Dropout`: Prevent overfitting
- `Softmax`: Output probabilities for each class

## ⚙️ Setup

pip install tensorflow keras matplotlib numpy pandas 

## 🧪 Applications
Medical diagnostics

Radiology automation

Educational demos for CNNs

Healthcare AI research

## 🤝 Contributing
Contributions are welcome! You can:

Add new tumor classes

Improve model architecture

Enhance preprocessing

Share performance benchmarks

### To contribute:

git clone https://github.com/yourusername/brain-tumor-mri-classifier.git

cd brain-tumor-mri-classifier

Submit a pull request with your improvements.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
