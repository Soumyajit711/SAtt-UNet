import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ISIC2016Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(224, 224), transform=None, mode='train', augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        self.augment = augment
        
        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        
        # For training/validation, we expect matching mask files
        if mode != 'test':
            self.mask_files = sorted([f.replace('.jpg', '_Segmentation.png') for f in self.image_files])
        else:
            self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Apply augmentations if in training mode and augment is True
        if self.mode == 'train' and self.augment:
            # Define augmentations here or in init. Using init is better generally but for this snippet I'll check what was in notebook.
            # Notebook defined self.augmentations in init.
            
            # Convert to PIL Image for augmentations (already are PIL)
            # Need to apply same seed for image and mask
            seed = torch.random.seed()
            
            if hasattr(self, 'augmentations'):
               torch.manual_seed(seed)
               image = self.augmentations(image)
               
               torch.manual_seed(seed)
               mask = self.augmentations(mask)
            
        else:
            # Just resize if no augmentation
            image = image.resize(self.image_size)
            mask = mask.resize(self.image_size)
        
        # Convert to numpy arrays
        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            # Convert to tensors
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            
        return image, mask

    # Add back the augmentations init part I missed in copy above
    def __init__(self, images_dir, masks_dir, image_size=(224, 224), transform=None, mode='train', augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        self.augment = augment
        
        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        
        # For training/validation, we expect matching mask files
        if mode != 'test':
            self.mask_files = sorted([f.replace('.jpg', '_Segmentation.png') for f in self.image_files])
        else:
            self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])

        # Define augmentations
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ])
