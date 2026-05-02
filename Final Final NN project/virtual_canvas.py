import cv2
import numpy as np

class VirtualCanvas:
    def __init__(self, width=1200, height=800):  # Larger default canvas size
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.canvas.fill(255)  # White background
        self.last_point = None
        self.line_thickness = 2
        self.line_color = (0, 0, 0)  # Black color for drawing
        self.current_stroke = []  # Store points for the current stroke
        self.strokes = []  # Store all completed strokes

    def start_stroke(self, point):
        """Begin a new stroke at the given point."""
        if point is None:
            return
        self.current_stroke = [point]
        self.last_point = point

    def end_stroke(self):
        """End the current stroke and save it."""
        if self.current_stroke:
            self.strokes.append(self.current_stroke)
            self.current_stroke = []
        self.last_point = None  # Reset last point to ensure the next stroke starts independently

    def draw(self, point, drawing=True):
        """Draw a line from the last point to the current point on the canvas."""
        if point is None:
            self.last_point = None
            return

        # Ensure point coordinates are within canvas bounds
        x = min(max(point[0], 0), self.width - 1)
        y = min(max(point[1], 0), self.height - 1)
        current_point = (x, y)

        if drawing:
            # Draw a circle at the current point
            cv2.circle(self.canvas, current_point, self.line_thickness, self.line_color, -1)

            # If we have a last point, connect it to the current point
            if self.last_point is not None:
                cv2.line(self.canvas, self.last_point, current_point, self.line_color, self.line_thickness)

            # Add point to the current stroke if the last point exists or it's the start of a new stroke
            if self.last_point is None or self.current_stroke:
                self.current_stroke.append(current_point)

        # Update the last point only if we're drawing
        if drawing:
            self.last_point = current_point
        else:
            self.last_point = None


    def clear(self):
        """Clear the canvas and reset all strokes."""
        self.canvas.fill(255)
        self.last_point = None
        self.current_stroke = []
        self.strokes = []

    def save(self, filename="equation.png"):
        """Save the current canvas to a file."""
        cv2.imwrite(filename, self.canvas)

    def get_canvas(self):
        """Return a copy of the current canvas."""
        return self.canvas.copy()

    def get_strokes(self):
        """Return all completed strokes."""
        return self.strokes
