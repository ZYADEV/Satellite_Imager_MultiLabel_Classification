# Satellite Imager MultiLabel Classification

A machine learning application for multi-label classification of satellite images using deep learning techniques. This project provides an interactive Streamlit web interface for uploading and classifying satellite imagery.

## 🚀 Features

- **Multi-label Classification**: Classify satellite images with multiple labels simultaneously
- **Interactive Web Interface**: User-friendly Streamlit application for easy image upload and prediction
- **Real-time Predictions**: Get instant classification results for uploaded satellite images
- **Deep Learning Model**: Powered by state-of-the-art neural networks for accurate predictions

## 🛠️ Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/ZYADEV/Satellite_Imager_MultiLabel_Classification.git
cd Satellite_Imager_MultiLabel_Classification
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
Satellite_Imager_MultiLabel_Classification/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── model_vf.h5            # Trained model file
├── notebook.ipynb            # Jupyter Notebook
```

## 🔧 Usage

1. **Upload Image**: Use the file uploader to select a satellite image
2. **Classification**: The model will automatically process and classify the image
3. **Results**: View the predicted labels with confidence scores

## 🤖 Model Information

- **Architecture**: Deep Convolutional Neural Network
- **Task**: Multi-label Image Classification
- **Input**: Satellite imagery (RGB/Multispectral)
- **Output**: Multiple classification labels with confidence scores

## 📊 Dataset

The model is trained on satellite imagery datasets with multiple environmental and geographical labels including:
- Land use classification
- Vegetation types
- Urban/Rural areas
- Water bodies
- Agricultural zones
- etc ....

## 👤 Author

**ZYADEV**
- GitHub: [@ZYADEV](https://github.com/ZYADEV)