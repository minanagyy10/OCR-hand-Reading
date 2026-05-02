import cv2
import numpy as np
from sklearn.preprocessing import normalize

class SymbolRecognizer:
    def __init__(self):
        self.templates = self._create_operator_templates()

    def _create_operator_templates(self):
        """Create template images for operators"""
        templates = {}

        # Create plus template
        plus = np.zeros((28, 28), dtype=np.uint8)
        cv2.line(plus, (14, 7), (14, 21), 255, 2)  # Vertical line
        cv2.line(plus, (7, 14), (21, 14), 255, 2)  # Horizontal line
        templates['+'] = plus

        # Create minus template
        minus = np.zeros((28, 28), dtype=np.uint8)
        cv2.line(minus, (7, 14), (21, 14), 255, 2)  # Horizontal line
        templates['-'] = minus

        return templates

    def preprocess_symbol(self, image):
        """Preprocess symbol image for recognition"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)

        # Resize to match template size
        resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)

        # Apply morphological operations to clean up noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def calculate_symbol_features(self, image):
        """Calculate features for symbol classification"""
        # Calculate aspect ratio
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        x, y, w, h = cv2.boundingRect(contours[0])
        aspect_ratio = w / h if h != 0 else 0

        density = np.sum(image > 0) / (image.shape[0] * image.shape[1])

        # Calculate symmetry scores
        vertical_symmetry = self._calculate_symmetry(image, 'vertical')
        horizontal_symmetry = self._calculate_symmetry(image, 'horizontal')

        return {
            'aspect_ratio': aspect_ratio,
            'density': density,
            'vertical_symmetry': vertical_symmetry,
            'horizontal_symmetry': horizontal_symmetry
        }

    def _calculate_symmetry(self, image, axis='vertical'):
        """Calculate symmetry score along specified axis"""
        if axis == 'vertical':
            left = image[:, :image.shape[1]//2]
            right = cv2.flip(image[:, image.shape[1]//2:], 1)
            return np.sum(left == right) / left.size
        else:
            top = image[:image.shape[0]//2, :]
            bottom = cv2.flip(image[image.shape[0]//2:, :], 0)
            return np.sum(top == bottom) / top.size

    def match_template(self, image, operator):
        """Match image against operator template"""
        template = self.templates[operator]
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        return np.max(result)

    def is_operator(self, image):
        """Determine if image contains an operator"""
        preprocessed = self.preprocess_symbol(image)
        features = self.calculate_symbol_features(preprocessed)

        if features is None:
            return False, None

        # Plus sign detection
        if (0.7 < features['aspect_ratio'] < 1.3 and
            0.05 < features['density'] < 0.4 and
            features['vertical_symmetry'] > 0.7 and
            features['horizontal_symmetry'] > 0.7):
            plus_score = self.match_template(preprocessed, '+')
            if plus_score > 0.01:  # Adjusted threshold
                return True, '+'

        # Minus sign detection
        if (features['aspect_ratio'] > 2.0 and
            features['density'] < 0.2 and
            features['horizontal_symmetry'] > 0.7):
            minus_score = self.match_template(preprocessed, '-')
            if minus_score > 0.01:  # Adjusted threshold
                return True, '-'

        return False, None

# Example usage (for testing):
if __name__ == "__main__":
    recognizer = SymbolRecognizer()

    # Test with a drawn plus sign
    test_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(test_image, (50, 20), (50, 80), 255, 5)  # Vertical line
    cv2.line(test_image, (20, 50), (80, 50), 255, 5)  # Horizontal line

    preprocessed = recognizer.preprocess_symbol(test_image)
    is_op, operator = recognizer.is_operator(preprocessed)
    print("Detected operator:", operator if is_op else "None")

    # Visualize for debugging
    cv2.imshow("Test Image", test_image)
    cv2.imshow("Preprocessed", preprocessed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
