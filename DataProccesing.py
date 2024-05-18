import numpy as np
from PIL import Image
from skimage.transform import resize
import os
from pympler.tracker import SummaryTracker

tracker = SummaryTracker()

def preprocess_image_numpy(image, width=64, height=48):
    # Convert to grayscale and crop the line for death detection
    line = np.array(image.convert('L').crop((0, 237, 256, 238)))

    # Sum the pixel values of the line to check for death condition
    # Assuming the death detection logic is based on summing the pixel values
    line_sum = np.sum(line)

    # Crop and resize the main image
    image = np.array(image.convert('L').crop((0, 0, 256, 192)))
    image_resized = resize(image, (height, width), anti_aliasing=True, mode='reflect')

    # Normalize the image
    image_normalized = image_resized / 255.0

    # Check if the player is dead
    dead = line_sum == 44044  # This condition might need adjustment

    return image_normalized, dead

def data():
    os.system('cls')
    tracker.print_diff()  
