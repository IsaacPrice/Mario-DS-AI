from PIL import Image
import numpy as np

def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    line = image.crop(())
    image = image.crop((0, 0, 256, 192))  # Crop
    image = image.resize((84, 84), Image.ANTIALIAS)  # Resize
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize
    return image
