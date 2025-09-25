# Import necessary libraries
import streamlit as st  # For building the web app UI
import cv2              # OpenCV for image processing
import numpy as np      # For numerical operations
from keras.models import load_model  # To load the trained deep learning model
from keras.preprocessing.image import img_to_array  # To preprocess image
from PIL import Image   # For image file handling
from camera import VideoCamera  # Import the VideoCamera class from camera.py

# Load the pre-trained emotion detection model (without compiling)
model = load_model("../emotion_model.hdf5", compile=False)

# Define the list of emotion labels corresponding to model predictions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect emotion in the uploaded image
def detect_emotion(image):
    # Convert PIL image to RGB NumPy array
    image = np.array(image.convert('RGB'))
    
    # Convert the RGB image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # If no faces found, return message and original image
    if len(faces) == 0:
        return "No face detected", image

    labels = []
    # For each detected face
    for (x, y, w, h) in faces:
        # Crop the face region from grayscale image
        roi_gray = gray[y:y+h, x:x+w]

        # Resize the face to match model input size
        roi = cv2.resize(roi_gray, (64, 64))

        # Normalize pixel values and prepare for prediction
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=-1) # Add channel dimension
        roi = np.expand_dims(roi, axis=0)  # Add batch dimension
        
        # Predict emotion using the model
        preds = model.predict(roi)[0]
        label = emotions[preds.argmax()] # Get emotion with highest probability
        labels.append(label)
  
        # Draw bounding box and emotion label on the original image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Join all detected emotions for display
    return ", ".join(labels), image

# Streamlit UI code
st.title("😊 Real Time Face Emotion Detector")  # App title

# Option to use camera for real-time detection
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

def run_camera():
    camera = VideoCamera()
    stop_button = st.button("Stop Camera", key="stop_camera")
    frame_placeholder = st.empty()
    while st.session_state.camera_running:
        frame = camera.get_frame()
        if frame is not None:
            labels, result_img = detect_emotion(frame)
            frame_placeholder.image(result_img, caption="Camera Preview", use_column_width=True)
        else:
            frame_placeholder.warning("Unable to access camera.")
        import time
        time.sleep(0.1)
        if stop_button:
            st.session_state.camera_running = False
            frame_placeholder.empty()  # Clear camera preview
            st.experimental_rerun()    # Refresh UI to show upload section
            break

if st.button("Use Camera", key="use_camera"):
    st.session_state.camera_running = True

if st.session_state.camera_running:
    run_camera()
else:
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        labels, result_img = detect_emotion(image)
        st.image(result_img, caption="Emotion Preview", use_column_width=True)
