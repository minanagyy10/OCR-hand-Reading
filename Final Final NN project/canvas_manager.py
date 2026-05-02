import tkinter as tk

class CanvasManager:
    def __init__(self, root):
        self.canvas = tk.Canvas(root, width=400, height=400, bg='white')
        self.canvas.pack(pady=20)
        
        self.drawing_enabled = True
        self.last_x = None
        self.last_y = None
        self.drawing_data = []
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
    
    def start_drawing(self, event):
        if not self.drawing_enabled:
            return
        self.last_x = event.x
        self.last_y = event.y
        self.drawing_data.append([(event.x, event.y)])
    
    def draw(self, event):
        if not self.drawing_enabled:
            return
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, 
                                  event.x, event.y, width=2)
            self.drawing_data[-1].append((event.x, event.y))
            self.last_x = event.x
            self.last_y = event.y
    
    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing_data = []
    
    def set_drawing_enabled(self, enabled):
        self.drawing_enabled = enabled
    
    def get_drawing_data(self):
        return self.drawing_data