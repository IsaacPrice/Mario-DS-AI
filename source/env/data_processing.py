import numpy as np
from PIL import Image
from skimage.transform import resize
import os

# The death is determined if marios marker in the screen below is showing or not
# since there is a flag that drops below the screen when mario passes half the level, there are a range of values where mario is not dead
acceptable_values = [340384, 340624, 340912, 343718, 341731, 342497, 340397, 340489, 
                     340437, 341367, 342297, 342387, 342471, 342798, 342406, 340332]

def preprocess_image(image, width=96, height=64, contrast_factor=3.0):
    """
    Enhanced PPO preprocessing with higher resolution and better feature extraction for platform detection
    """
    gray_image = image.convert('L')
    
    image_array = np.array(gray_image)
    image_contrasted = np.clip((image_array.astype(np.float32) / 255.0 - 0.5) * contrast_factor + 0.5, 0, 1)
    image_contrasted = (image_contrasted * 255).astype(np.uint8)
    
    image_enhanced = Image.fromarray(image_contrasted)
    
    line = np.array(gray_image.crop((0, 237, 256, 245)))
    line_sum = np.sum(line)
    dead = line_sum not in acceptable_values
    
    image_cropped = np.array(image_enhanced.crop((0, 0, 256, 192)))
    image_normalized = resize(image_cropped, (height, width), anti_aliasing=True, mode='reflect') / 255.0

    return image_normalized, dead

def data():
    os.system('cls')
