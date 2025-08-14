import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the custom fbeta function
def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2

    # Clipping y_pred between 0 and 1
    y_pred = K.clip(y_pred, 0, 1)

    # Rounding y_pred to binary values
    y_pred_bin = K.round(y_pred + threshold_shift)

    # Counting true positives, false positives, and false negatives
    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    # Calculating precision and recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    # Calculating the F-beta score
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

# Define the custom accuracy_score function
def accuracy_score(y_true, y_pred, epsilon=1e-4):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.greater(tf.cast(y_pred, tf.float32), tf.constant(0.5)), tf.float32)

    tp = tf.reduce_sum(y_true * y_pred, axis=1)
    fp = tf.reduce_sum(y_pred, axis=1) - tp
    fn = tf.reduce_sum(y_true, axis=1) - tp

    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    tn = tf.reduce_sum(tf.cast(tf.logical_not(y_true), tf.float32) * tf.cast(tf.logical_not(y_pred), tf.float32), axis=1)

    return (tp + tn) / (tp + tn + fp + fn + epsilon)

# Register custom objects globally
tf.keras.utils.get_custom_objects().update({'fbeta': fbeta, 'accuracy_score': accuracy_score})

# Load the model with error handling
@st.cache_resource
def load_model():
    try:
        model_filename = 'model_vf.h5'
        loaded_model = keras.models.load_model(
            model_filename, 
            custom_objects={'fbeta': fbeta, 'accuracy_score': accuracy_score},
            compile=False
        )
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

loaded_model = load_model()

# Function to preprocess a single image
def preprocess_image(image_path):
    # Read the image file
    img = cv2.imread(image_path)

    # Resize the image to (64, 64)
    img = cv2.resize(img, (128, 128))

    # Normalize pixel values to the range [0, 1]
    img = img.astype(np.float16) / 255.0

    return img

# Function to make predictions on a single image
def predict_single_image(model, img):
    # Reshape the image to match the model's input shape
    img = np.reshape(img, (1, 128, 128, 3))

    # Make predictions on the single image
    prediction = model.predict(img)

    # Apply a threshold of 0.5 for binary predictions
    prediction_binary = (prediction >= 0.1).astype(int)

    return prediction_binary

def main():
    st.title("Satellite Image Multi-Classification with CNN")
    
    if loaded_model is None:
        st.error("Failed to load the model. Please check if the model file exists and is valid.")
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file
        filename = "uploaded_image.jpg"
        with open(filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Preprocess the image
        img = preprocess_image(filename)

        # Make predictions on the single image
        predictions_single_image = predict_single_image(loaded_model, img)
        # Define the class names
        classes = ['haze', 'primary', 'agriculture', 'blooming','clear', 'habitation',
                   'water', 'cultivation', 'slash_burn', 'cloudy', 'partly_cloudy',
                   'conventional_mine', 'bare_ground', 'artisanal_mine', 'road',
                   'selective_logging', 'blow_down']

        # Get the class names with icons
        class_icons = {
            'haze': '🌫️', 'primary': '🌲', 'agriculture': '🚜', 'clear': '☀️',
            'water': '💧', 'habitation': '🏠', 'road': '🛣️', 'cultivation': '🌾',
            'slash_burn': '🔥', 'cloudy': '☁️', 'partly_cloudy': '⛅',
            'conventional_mine': '⛏️', 'bare_ground': '🏜️', 'artisanal_mine': '⚒️',
            'blooming': '🌸', 'selective_logging': '🪓', 'blow_down': '🌪️'
        }

        # Map numerical labels to class names
        predicted_classes = [classes[i] for i, label in enumerate(predictions_single_image[0]) if label == 1]

        # Display icons and titles for predicted classes
        if predicted_classes:
            st.subheader("Predicted Classes:")
            for predicted_class in predicted_classes:
                st.write(f"{class_icons[predicted_class]} {predicted_class}")
        else:
            st.write("No classes detected with the current threshold.")

        # Plot the image with predicted classes
        img = mpimg.imread(filename)
        st.image(img, caption="Uploaded Image", use_column_width=True)

if __name__ == "__main__":
    main()