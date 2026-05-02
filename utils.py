import cv2
import numpy as np

def resize_with_aspect_ratio(image, width=None, height=None):
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
        
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim)

def combine_images(images, direction='horizontal'):
    if direction == 'horizontal':
        return np.hstack(images)
    else:
        return np.vstack(images)