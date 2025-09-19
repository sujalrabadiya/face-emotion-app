# Face Emotion Detector

A deep learning project that classifies human emotions from uploaded images. The application is built with Streamlit, providing a simple and intuitive user interface.

## Table of Contents
- [About the Project](#about-the-project)
- [How It Works](#how-it-works)
- [Live Demo](#live-demo)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Technologies](#technologies)
- [Acknowledgments](#acknowledgments)

## About the Project

This project leverages the power of Convolutional Neural Networks (CNNs) to accurately detect and classify facial emotions. The model is trained on the **FER-2013 dataset**, which includes thousands of grayscale facial images labeled with one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

The core of the project is the trained deep learning model, which is integrated into a Streamlit web application. Users can upload an image, and the application will detect any faces in the image and predict the corresponding emotion.

## How It Works

1.  **Image Upload**: A user uploads an image file through the Streamlit interface.
2.  **Face Detection**: The application uses a pre-trained Haar Cascade classifier to identify faces within the uploaded image.
3.  **Preprocessing**: Each detected face is converted to grayscale, resized to $48 \times 48$ pixels, and prepared for model input.
4.  **Emotion Prediction**: The preprocessed image is fed into the trained CNN model, which outputs a prediction for one of the seven emotions.
5.  **Result Display**: The original image is displayed with a bounding box around the detected face and a text label indicating the predicted emotion.

## Live Demo

You can try the live application here:

[https://face-emotion-app-sujalrabadiya.streamlit.app/](https://face-emotion-app-sujalrabadiya.streamlit.app/)

## Installation

To run this project locally, follow these steps:

1.  Clone the repository:
    ```bash
    git clone [https://github.com/sujalrabadiya/face-emotion-app.git](https://github.com/sujalrabadiya/face-emotion-app.git)
    cd face-emotion-app
    ```

2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once you have installed the dependencies, you can run the Streamlit application from your terminal:

```bash
streamlit run app.py
````

## Model Details

The model is a sequential CNN built using TensorFlow and Keras.

  - **Architecture**:
      - Multiple $Conv2D$, $MaxPooling2D$, and Dropout layers.
      - A Flatten layer.
      - A final Dense layer with a **softmax** activation function to output the emotion probabilities.
  - **Dataset**: **FER-2013** (Facial Expression Recognition 2013)
  - **Training**:
      - **Epochs**: 50
      - **Training Accuracy**: 59.16%
      - **Validation Accuracy**: 64.02%

## Technologies

  - **Python**
  - **TensorFlow / Keras**: For building and training the deep learning model.
  - **Streamlit**: For creating the web application.
  - **OpenCV**: For image preprocessing and face detection.
  - **NumPy**: For numerical operations.

## Acknowledgments

  - This project was developed under the guidance of Prof. Jayesh D Vagadiya.
  - The training was performed using Google Colab.

<!-- end list -->
