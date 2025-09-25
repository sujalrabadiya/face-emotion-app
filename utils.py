# Utility functions for image processing

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

def preprocess_image(image):
    """Convert image to grayscale, resize, and normalize for model input."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to match model input size
    roi = cv2.resize(gray, (64, 64))
    
    # Normalize pixel values
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=-1)  # Add channel dimension
    roi = np.expand_dims(roi, axis=0)   # Add batch dimension
    
    return roi

def draw_label(image, label, position):
    """Draw bounding box and label on the image."""
    cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
