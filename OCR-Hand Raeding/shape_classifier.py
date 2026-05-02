import numpy as np
import cv2
from tensorflow.keras import layers, models

class ShapeClassifier:
    def __init__(self):
        self.classes = ['circle', 'rectangle', 'square', 'triangle']
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(64, 64, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.classes), activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def preprocess_image(self, image):
        if len(image.shape) == 3:  # Convert to grayscale if needed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64, 64))
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)   # Add batch dimension
        return image

    def calculate_shape_properties(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            return area, perimeter, cnt
        return 0, 0, None

    def contour_shape_detection(self, cnt):
        # Approximate the contour
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        vertices = len(approx)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            # Check aspect ratio for rectangle vs square
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                return "square"
            else:
                return "rectangle"
        elif vertices > 4:
            # Additional check for circles using area-to-perimeter ratio
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if 0.8 <= circularity <= 1.2:
                return "circle"

        # If no match, return unknown
        return "unknown"

    def predict_shape(self, image):
        processed_img = self.preprocess_image(image)
        prediction = self.model.predict(processed_img)
        shape_class = self.classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Fallback using contour analysis
        area, perimeter, cnt = self.calculate_shape_properties(image)
        if cnt is not None:
            contour_shape = self.contour_shape_detection(cnt)
            if contour_shape != shape_class:
                print(f"Model predicted: {shape_class}, Contour analysis: {contour_shape}")
                shape_class = contour_shape  # Use contour analysis if different

        return {
            'shape': shape_class,
            'confidence': confidence,
            'area_pixels': area,
            'perimeter_pixels': perimeter
        }
