import torch
from torchvision import transforms
from PIL import Image

def preprocess_image_tensor(image, width=128, height=96):
    # Convert to grayscale and crop the line for death detection
    line = image.convert('L').crop((0, 253, 256, 254))

    # Convert line to tensor and normalize
    line_tensor = transforms.ToTensor()(line)
    line_tensor = torch.sum(line_tensor)

    # Crop and resize the main image
    image = image.convert('L').crop((0, 0, 256, 192))
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image)

    # Normalize the image tensor
    image_tensor = image_tensor / 255.0

    # Check if the player is dead
    dead = line_tensor.item() == 172.09019470214844

    return image_tensor, dead