# 🧠 Brain Tumor MRI Classifier

This project uses a Convolutional Neural Network (CNN) to classify brain tumors from MRI images. It leverages TensorFlow and Keras to build, train, and evaluate a deep learning model capable of distinguishing between different tumor types.

## 📂 Dataset

The dataset should be organized in the following structure:

Brain Tumor/ 
└── Training/ 
├── TumorType1/ 
├── TumorType2/ 
└── TumorType3/

Each subfolder under `Training` should contain MRI images corresponding to a specific tumor class.

## 🧪 Model Architecture

- Input: 64x64 RGB MRI images
- Layers:
  - Rescaling
  - Conv2D + ReLU
  - MaxPooling
  - Conv2D + ReLU
  - MaxPooling
  - Flatten
  - Dense + ReLU
  - Dropout
  - Dense + Softmax (output)

## ⚙️ Setup

pip install tensorflow matplotlib

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
