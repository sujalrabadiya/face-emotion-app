import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

from PIL import Image

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 0 is the default camera

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Convert OpenCV BGR image to PIL RGB image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
            
# Load the pre-trained emotion detection model
model = load_model("../emotion_model.hdf5", compile=False)

# Define the list of emotion labels corresponding to model predictions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion_from_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # If no faces found, return original frame
    if len(faces) == 0:
        return frame, "No face detected"

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
        roi = np.expand_dims(roi, axis=-1)  # Add channel dimension
        roi = np.expand_dims(roi, axis=0)   # Add batch dimension
        
        # Predict emotion using the model
        preds = model.predict(roi)[0]
        label = emotions[preds.argmax()]  # Get emotion with highest probability
        labels.append(label)

        # Draw bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Join all detected emotions for display
    return frame, ", ".join(labels)

def start_camera():
    # Start video capture from the camera
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect emotion in the current frame
        result_frame, emotions_detected = detect_emotion_from_frame(frame)

        # Display the resulting frame with detected emotions
        cv2.imshow('Real-time Face Emotion Detection', result_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()