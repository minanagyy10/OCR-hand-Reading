import cv2
from hand_tracker import HandTracker
from virtual_canvas import VirtualCanvas
from shape_classifier import ShapeClassifier
from equation_recognizer import EquationRecognizer
from model.cnn_model import train_model
import os

def main():
    # Train model if it doesn't exist
    if not os.path.exists('mnist_model.h5'):
        print("Training MNIST model...")
        train_model()
        print("Model training completed!")

    # Initialize components
    cap = cv2.VideoCapture(0)
    hand_tracker = HandTracker()
    canvas = VirtualCanvas(width=1200, height=800)  # Larger canvas
    recognizer = EquationRecognizer()
    shape_classifier = ShapeClassifier()
    shape_classifier.model.load_weights("shape_classifier_weights.h5")  # Load pre-trained weights

    drawing_mode = False
    last_result = None
    is_drawing = False  # Track if currently drawing a stroke

    print("Controls:")
    print("Press 'd' to toggle drawing mode")
    print("Press 'c' to clear canvas")
    print("Press 's' to save and recognize equation")
    print("Press 'h' to detect shapes on canvas")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame = cv2.flip(frame, 1)  # Flip for intuitive interaction
        finger_point, hand_detected = hand_tracker.get_index_finger_tip(frame)
        finger_height = hand_tracker.get_finger_height(frame) if hand_detected else None

        # Handle drawing
        if drawing_mode and hand_detected:
            if finger_height and finger_height < 0.7:  # Finger raised
                if not is_drawing:
                    canvas.start_stroke(finger_point)
                    is_drawing = True
                canvas.draw(finger_point)  # Continue drawing
                cv2.circle(frame, finger_point, 5, (0, 255, 0), -1)  # Visual feedback
            elif is_drawing:  # End stroke
                canvas.end_stroke()
                is_drawing = False

        # Display the canvas and frame
        canvas_display = canvas.get_canvas()
        frame = cv2.resize(frame, (canvas.width, canvas.height))  # Match sizes

        # Display modes and results
        mode_text = "Drawing Mode: ON" if drawing_mode else "Drawing Mode: OFF"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if last_result:
            cv2.putText(frame, last_result, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Camera Feed', frame)
        cv2.imshow('Virtual Canvas', canvas_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            print("Exiting...")
            break
        elif key == ord('c'):  # Clear canvas
            canvas.clear()
            last_result = None
            print("Canvas cleared!")
        elif key == ord('h'):  # Detect shapes
            try:
                canvas_image = canvas.get_canvas()
                result = shape_classifier.predict_shape(canvas_image)
                print("\nShape Detection Results:")
                print(f"Shape: {result['shape']}")
                print(f"Confidence: {result['confidence']:.2f}%")
                print(f"Area: {result['area_pixels']:.2f}")
                print(f"Perimeter: {result['perimeter_pixels']:.2f}")
            except Exception as e:
                print(f"Error detecting shape: {e}")
        elif key == ord('d'):  # Toggle drawing mode
            drawing_mode = not drawing_mode
            print(f"Drawing mode: {'ON' if drawing_mode else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
