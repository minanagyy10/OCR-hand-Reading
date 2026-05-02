import cv2
import numpy as np
import tensorflow as tf
from .symbol_recognizer import SymbolRecognizer

class DigitRecognizer:
    def __init__(self, model_path='mnist_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.symbol_recognizer = SymbolRecognizer()

    def preprocess_image(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to 28x28
        image = cv2.resize(image, (28, 28))

        # Normalize
        image = image.astype('float32') / 255.0

        # Reshape for model input
        image = image.reshape(1, 28, 28, 1)
        return image

    def predict(self, image):
        """Predict if the image is a digit or operator."""
        is_op, op_type = self.symbol_recognizer.is_operator(image)
        if is_op:
            return op_type

        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image)
        digit = str(np.argmax(prediction[0]))
        
        # Log the prediction
        print(f"Predicted digit: {digit}")
        
        return digit
