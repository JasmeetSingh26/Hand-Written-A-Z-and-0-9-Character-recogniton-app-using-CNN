import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(32, (5, 5), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dense(36, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Load the model weights
model.load_weights("models/best_val_loss_model.h5")

# Function to preprocess the image
def preprocess_image(uploaded_file):
    # Convert the uploaded file to a PIL Image
    pil_image = Image.open(uploaded_file).convert('L')  # 'L' mode for grayscale

    # Resize the image to (28, 28)
    pil_image = pil_image.resize((28, 28))

    plt.imshow(np.array(pil_image), cmap='gray')
    plt.show()

    img_array = img_to_array(pil_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    return img_array

# Function to get the model prediction
def get_prediction(img_array):
    prediction = model.predict(img_array)
    return np.argmax(prediction)

def main():
    st.title("Character Recognition Web App")
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(uploaded_file)

        # Get the prediction
        prediction = get_prediction(img_array)
        class_label = labels[prediction]

        st.write(f"Predicted Character: {class_label}")

if __name__ == "__main__":
    main()
