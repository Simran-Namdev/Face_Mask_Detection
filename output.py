import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

def predict_on_frame(frame, model):
    # Define image dimensions
    img_width, img_height = 200, 200

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    for (x, y, w, h) in faces:
        # Extract the face ROI
        color_face = frame[y:y + h, x:x + w]

        # Preprocess the face image for the model
        img = cv2.resize(color_face, (img_width, img_height))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize pixel values

        # Make predictions using the model
        predictions = model.predict(img)

        # Interpret predictions
        if predictions[0][0] < 0.5:
            class_label = "Mask"
            color = (0, 255, 0)  # Green for mask
        else:
            class_label = "No Mask"
            color = (0, 0, 255)  # Red for no mask

        # Draw rectangle around the face and add label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

def detect_face_mask(video_source, model):
    cap = cv2.VideoCapture(video_source)

    # Create a button to stop the live video feed
    stop_button = st.button("Stop Live Video")

    while cap.isOpened() and not stop_button:
        response, color_img = cap.read()

        if not response:
            break

        # Process each frame for face mask detection
        processed_frame = predict_on_frame(color_img, model)

        # Display the processed frame
        st.image(processed_frame, channels="BGR", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()

def predict_on_image(image_path):
    # Load the pre-trained model
    model = load_model('save_m0.h5')

    # Load and preprocess the image
    img_width, img_height = 200, 200
    image = Image.open(image_path)
    image = image.resize((img_width, img_height))
    input_arr = img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch.

    # Normalize pixel values between 0 and 1
    input_arr = input_arr / 255.0

    # Make predictions using the model
    predictions = model.predict(input_arr)

    # Interpret predictions
    if predictions[0] < 0.5:
        return "Mask is present"
    else:
        return "Mask is not present"

def main():
    st.title("Face Mask Detection")
    model = load_model('save_m0.h5')

    option = st.radio("Choose option", ("Live Video Capture", "Upload Picture"))

    if option == "Live Video Capture":
        st.write("Click 'Start' to begin live video capture.")
        if st.button("Start"):
            detect_face_mask(0, model)

    elif option == "Upload Picture":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.write("Uploaded image:")
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_column_width=True)
            if st.button('Submit'):
                result = predict_on_image(uploaded_file)
                st.write(result)

if __name__ == "__main__":
    main()
