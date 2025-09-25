# Real Time Face Emotion Detection App

This project is a web application that detects emotions from faces using a pre-trained deep learning model. Users can upload images or use their camera for real-time emotion detection.

## Project Structure

```
face-emotion-app
├── src
│   ├── app.py                    # Main entry point for the Streamlit web app
│   ├── camera.py                 # Handles real-time video capture and emotion detection
│   └── utils.py                  # Utility functions for image processing
├── emotion_model.hdf5            # Pre-trained deep learning model for emotion detection
├── emotion_model_training.ipynb  # Notebook for trainning emotion_model.hdf5
├── requirements.txt              # Python dependencies for the project
└── README.md                     # Documentation for the project
```

## Table of Contents
- [About the Project](#about-the-project)
- [How It Works](#how-it-works)
- [Live Demo](#live-demo)
- [Setup Instructions](#setup-instructions)
- [Usage Guidelines](#usage-guidelines)
- [Model Details](#model-details)
- [Dependencies](#dependencies)

## About the Project

This project leverages the power of Convolutional Neural Networks (CNNs) to accurately detect and classify facial emotions. The model is trained on the **FER-2013 dataset**, which includes thousands of grayscale facial images labeled with one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

The core of the project is the trained deep learning model, which is integrated into a Streamlit web application. Users can upload an image or use their camera, and the application will detect any faces in the image and predict the corresponding emotion.

## How It Works

1.  **Image Upload/Use Camera**: A user uploads an image file through the Streamlit interface.
2.  **Face Detection**: The application uses a pre-trained Haar Cascade classifier to identify faces within the uploaded image.
3.  **Preprocessing**: Each detected face is converted to grayscale, resized to $64 \times 64$ pixels, and prepared for model input.
4.  **Emotion Prediction**: The preprocessed image is fed into the trained CNN model, which outputs a prediction for one of the seven emotions.
5.  **Result Display**: The original image is displayed with a bounding box around the detected face and a text label indicating the predicted emotion.

## Live Demo

You can try the live application here:

[https://face-emotion-app-sujalrabadiya.streamlit.app/](https://face-emotion-app-sujalrabadiya.streamlit.app/)

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone https://github.com/sujalrabadiya/face-emotion-app.git
   cd face-emotion-app
   ```

2. **Install the required packages**:
   Create a virtual environment (optional but recommended) and install the dependencies listed in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Start the Streamlit app by running:
   ```
   python -m streamlit run src/app.py
   ```

## Usage Guidelines

- **Upload an Image**: Users can upload a face image in JPG, JPEG, or PNG format to detect emotions.
- **Real-time Detection**: Users can also use their camera for real-time emotion detection by clicking the appropriate button in the app interface.

## Model Details

The model is a sequential CNN built using TensorFlow and Keras.

  - **Architecture**:
      - Multiple $Conv2D$, $MaxPooling2D$, and Dropout layers.
      - A Flatten layer.
      - A final Dense layer with a **softmax** activation function to output the emotion probabilities.
  - **Dataset**: **[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)** (Facial Expression Recognition 2013)
  - **Training**:
      - **Epochs**: 50
      - **Training Accuracy**: 59.16%
      - **Validation Accuracy**: 64.02%

## Dependencies

  - **Python**
  - **TensorFlow / Keras**: For building and training the deep learning model.
  - **Streamlit**: For creating the web application.
  - **OpenCV**: For image preprocessing and face detection.
  - **NumPy**: For numerical operations.

<!-- end list -->
