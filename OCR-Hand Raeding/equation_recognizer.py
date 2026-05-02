import cv2
import numpy as np
from recognition.digit_recognizer import DigitRecognizer
from recognition.equation_parser import EquationParser

class EquationRecognizer:
    def __init__(self):
        self.digit_recognizer = DigitRecognizer()
        self.equation_parser = EquationParser()
        self.min_symbol_spacing = 20  # Minimum pixels between symbols
        self.operator_symbols = {'+', '-', '*', '/'}

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # تحسين إزالة الضوضاء
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        return thresh

    def segment_equation(self, image):
        """Segment the equation into symbols based on spacing."""
        preprocessed = self.preprocess_image(image)
        
        # Find contours of all symbols
        contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 50:  # Ignore very small noise
                # Append the bounding box directly as a tuple (x, y, w, h)
                bounding_boxes.append((x, y, w, h))
        
        # Sort bounding boxes from left to right
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

        # Group symbols based on spacing
        final_boxes = []
        min_spacing = self.min_symbol_spacing
        previous_x = None
        
        for box in bounding_boxes:
            if previous_x is None or box[0] - previous_x > min_spacing:
                final_boxes.append(box)
            else:
                # Combine symbols too close to each other
                last_box = final_boxes.pop()
                combined_x = min(last_box[0], box[0])
                combined_w = max(last_box[0] + last_box[2], box[0] + box[2]) - combined_x
                final_boxes.append((combined_x, box[1], combined_w, box[3]))
            previous_x = box[0] + box[2]

        return final_boxes, preprocessed  # Return both final_boxes and preprocessed image




    def recognize_symbol(self, image, box):
        x, y, w, h = box
        symbol_img = image[y:y + h, x:x + w]
        
        # Ensure the symbol has content
        if np.sum(symbol_img) == 0:
            return ''
            
        # Calculate aspect ratio to help identify operators
        aspect_ratio = w / h if h > 0 else 0
        
        # Normalize symbol size
        normalized_img = cv2.resize(symbol_img, (28, 28))
        
        # Use aspect ratio and symbol position to improve operator detection
        if aspect_ratio < 0.8:  # Tall and narrow symbols might be operators
            # Additional checks for operator characteristics
            vertical_projection = np.sum(normalized_img, axis=0)
            horizontal_projection = np.sum(normalized_img, axis=1)
            
            # Check for plus sign characteristics
            if (max(vertical_projection) > 0 and max(horizontal_projection) > 0 and
                abs(max(vertical_projection) - max(horizontal_projection)) < 500):
                return '+'
                
        return self.digit_recognizer.predict(normalized_img)
    
    def validate_symbols(self, symbols):
        """
        تصفية الرموز للتأكد من أن جميعها صالحة ومعترف بها.
        :param symbols: قائمة الرموز الناتجة عن التعرف.
        :return: قائمة الرموز بعد التصفية.
        """
        valid_symbols = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '×', '/', '='}
        return [s for s in symbols if s in valid_symbols]

    def postprocess_results(self, results):
     
        # قائمة الرموز الصالحة
        valid_symbols = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '×', '/', '='}
        equation = []

        for symbol in results:
            # إذا كان الرمز في قائمة الرموز الصالحة، أضفه إلى المعادلة
            if symbol in valid_symbols:
                equation.append(symbol)
            else:
                # يمكنك إضافة خطوة للتعامل مع الرموز غير الصالحة (مثل تصحيحها أو تجاهلها)
                print(f"Invalid symbol detected: {symbol}")

        return ''.join(equation)

    def evaluate_equation(self, equation_str):
      
        try:
            # استخدام EquationParser لتحليل المعادلة
            result, message = self.equation_parser.parse_equation(equation_str)
            return result, message
        except Exception as e:
            return False, f"Error evaluating equation: {str(e)}"