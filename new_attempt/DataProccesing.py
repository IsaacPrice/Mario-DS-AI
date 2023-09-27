from PIL import Image
import numpy as np

def preprocess_image(image, width=64, height=48):
    image = image.convert('L')  # Convert to grayscale
    line = image.crop((0, 253, 256, 254)) # Add in the function where it will determine if the player has died or not
    # Add up all of the values inside of the line
    line = np.array(line)
    line = line / 255.0
    line = np.sum(line)
    image = image.crop((0, 0, 256, 192))  # Crop
    image = image.resize((width, height), Image.ANTIALIAS)  # Resize
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize
    dead = line == 172.09019607843138
    return image, dead
