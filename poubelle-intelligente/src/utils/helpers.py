def load_image(image_path):
    from PIL import Image
    import numpy as np

    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize the image
    return image_array

def preprocess_image(image_array):
    # Add any additional preprocessing steps if necessary
    return image_array

def decode_predictions(predictions, class_labels):
    decoded = {}
    for i, pred in enumerate(predictions):
        decoded[class_labels[i]] = pred
    return decoded

def load_class_labels(label_file):
    with open(label_file, 'r') as file:
        labels = file.read().splitlines()
    return labels