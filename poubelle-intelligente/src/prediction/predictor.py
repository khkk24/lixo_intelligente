from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2


class Predictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def preprocess_image(self, image):


        # Resize and scale the image
        image = load_img(image, target_size=(224, 224))  # Adjust size as per your model's requirement
        image = img_to_array(image)
        image = image / 255.0  # Normalize the image
        # image = image.reshape((1, *image.shape))  # Reshape for the model
        image = np.expand_dims(image, axis=0)
        return image
    
    
    def predict(self, image_path):
        processed_image = self.preprocess_image(image_path)
        prediction = self.model.predict(processed_image)
        return prediction[0][0]  # Return the class index with the highest probability

    def classify(self, image_path):
        prob = self.predict(image_path)
        class_index = 1 if prob >= 0.5 else 0
        return "Reciclável" if class_index == 1 else "Não reciclável"  # Adjust based on your class mapping
    
    
    
    def preprocess_image_pil(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            raise ValueError("Input must be a file path or a PIL Image object.")

        image = image.resize((224, 224))  # Do resizing here while it's still a PIL Image

        image = img_to_array(image)       # Convert to numpy array
        image = image / 255.0             # Normalize to [0,1]

        # Handle grayscale or alpha if still necessary
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel RGB image.")

        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image