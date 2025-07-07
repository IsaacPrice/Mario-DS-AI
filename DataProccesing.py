import numpy as np
from PIL import Image
from skimage.transform import resize
import os

def preprocess_image_numpy(image, width=64, height=48, contrast_factor=3):
    # Apply contrast enhancement to the original color image first
    image_array = np.array(image)
    image_contrasted = np.clip((image_array.astype(np.float32) / 255.0 - 0.5) * contrast_factor + 0.5, 0, 1)
    image_contrasted = (image_contrasted * 255).astype(np.uint8)
    image_enhanced = Image.fromarray(image_contrasted)
    
    # Convert to grayscale and crop the line for death detection
    line = np.array(image.convert('L').crop((0, 237, 256, 238)))

    # Sum the pixel values of the line to check for death condition
    line_sum = np.sum(line)

    # Crop and resize the main image
    image = np.array(image_enhanced.convert('L').crop((0, 0, 256, 192)))
    image_resized = resize(image, (height, width), anti_aliasing=True, mode='reflect')

    # Normalize the image for final output
    image_normalized = image_resized / 255.0

    # Check if the player is dead
    dead = line_sum == 44044  # This condition might need adjustment

    return image_normalized, dead

def preprocess_image_ppo(image, width=64, height=48, contrast_factor=2.5):
    """
    PPO-optimized image preprocessing with better feature extraction
    """
    # Convert to grayscale first
    gray_image = image.convert('L')
    
    # Apply contrast enhancement
    image_array = np.array(gray_image)
    image_contrasted = np.clip((image_array.astype(np.float32) / 255.0 - 0.5) * contrast_factor + 0.5, 0, 1)
    image_contrasted = (image_contrasted * 255).astype(np.uint8)
    
    # Create enhanced image
    image_enhanced = Image.fromarray(image_contrasted)
    
    # Check for death using line detection
    line = np.array(gray_image.crop((0, 237, 256, 238)))
    line_sum = np.sum(line)
    dead = line_sum == 44044
    
    # Crop and resize the main image
    image_cropped = np.array(image_enhanced.crop((0, 0, 256, 192)))
    image_resized = resize(image_cropped, (height, width), anti_aliasing=True, mode='reflect')
    
    # Normalize to [0, 1] range
    image_normalized = image_resized / 255.0
    
    return image_normalized, dead

def preprocess_image_ppo_enhanced(image, width=96, height=64, contrast_factor=3.0):
    """
    Enhanced PPO preprocessing with higher resolution and better feature extraction for platform detection
    """
    # Convert to grayscale first
    gray_image = image.convert('L')
    
    # Apply stronger contrast enhancement to better distinguish platforms
    image_array = np.array(gray_image)
    image_contrasted = np.clip((image_array.astype(np.float32) / 255.0 - 0.5) * contrast_factor + 0.5, 0, 1)
    image_contrasted = (image_contrasted * 255).astype(np.uint8)
    
    # Create enhanced image
    image_enhanced = Image.fromarray(image_contrasted)
    
    # Check for death using line detection
    line = np.array(gray_image.crop((0, 237, 256, 238)))
    line_sum = np.sum(line)
    dead = line_sum == 44044
    
    # Crop and resize the main image with higher resolution
    image_cropped = np.array(image_enhanced.crop((0, 0, 256, 192)))
    image_resized = resize(image_cropped, (height, width), anti_aliasing=True, mode='reflect')
    
    # Normalize to [0, 1] range
    image_normalized = image_resized / 255.0
    
    return image_normalized, dead

def data():
    os.system('cls')
