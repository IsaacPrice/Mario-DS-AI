import torch
import torch.nn as nn
import kornia.augmentation as K


class DrQv2Augmentation(nn.Module):
    def __init__(self, pad_size=4):
        super().__init__()
        self.pad_size = pad_size
        
        self.augmentations = nn.Sequential(
            K.RandomCrop(size=(64, 96), padding=pad_size),
            K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
        )
        

    def forward(self, frames):
        with torch.no_grad():
            augmented_frames = self.augmentations(frames)
            
        return augmented_frames
    
    
    def augment_multiple(self, frames, num_augmentations=2):
        augmented_frames_list = [frames]  # Include the original frames
        
        for _ in range(num_augmentations):
            augmented_frames = self.forward(frames)
            augmented_frames_list.append(augmented_frames)
            
        return augmented_frames_list


class PPODrQv2Augmentation(nn.Module):
    def __init__(self, num_augmentations=2):
        super().__init__()
        self.augmentor = DrQv2Augmentation()
        self.num_augmentations = num_augmentations
        
    
    def to(self, device):
        self.augmentor = self.augmentor.to(device)
        return self
        
    
    def augment_batch(self, frames):
        device = frames.device
        self.augmentor = self.augmentor.to(device)
        
        augmented_frames_list = self.augmentor.augment_multiple(frames, self.num_augmentations)
        
        augmented_batch = torch.cat(augmented_frames_list, dim=0)
        
        return augmented_batch
        