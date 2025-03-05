import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess_image(image_file):
    # Decode the image and convert it to the required format
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))  # ResNet50 expects 224x224 input images
    image = preprocess_input(image)        # Preprocess according to ResNet requirements
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
