class Predictor:
    def __init__(self, model_path):
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)

    def preprocess_image(self, image):
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.preprocessing.image import load_img
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Resize and scale the image
        image = load_img(image, target_size=(224, 224))  # Adjust size as per your model's requirement
        image = img_to_array(image)
        image = image / 255.0  # Normalize the image
        image = image.reshape((1, *image.shape))  # Reshape for the model
        return image

    def predict(self, image_path):
        processed_image = self.preprocess_image(image_path)
        prediction = self.model.predict(processed_image)
        return prediction.argmax(axis=1)[0]  # Return the class index with the highest probability

    def classify(self, image_path):
        class_index = self.predict(image_path)
        return "Recyclable" if class_index == 0 else "Non-Recyclable"  # Adjust based on your class mapping