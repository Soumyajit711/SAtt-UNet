import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from data.dataset import ISIC2016Dataset
from models.network import MSDUNet_MBV2_CorrCSA
from utils.loss import AdaptiveSegmentationLoss
from train import train_model

def get_args():
    parser = argparse.ArgumentParser(description="Train SleekNet on ISIC2016")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to training images")
    parser.add_argument("--masks_dir", type=str, required=True, help="Path to training masks")
    parser.add_argument("--val_images_dir", type=str, help="Path to validation images (optional, if separate)")
    parser.add_argument("--val_masks_dir", type=str, help="Path to validation masks (optional, if separate)")
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="Path to save best model")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split ratio if val directories not provided")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    train_dataset = ISIC2016Dataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        mode='train',
        augment=True
    )
    
    if args.val_images_dir and args.val_masks_dir:
        val_dataset = ISIC2016Dataset(
            images_dir=args.val_images_dir,
            masks_dir=args.val_masks_dir,
            mode='test',
            augment=False
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        # Random split
        full_size = len(train_dataset)
        val_size = int(args.validation_split * full_size)
        train_size = full_size - val_size
        
        train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
        
        print(f"Splitting dataset: {train_size} training, {val_size} validation")
        print("Warning: Validation set using random_split on a single dataset instance will retain augmentation if enabled in dataset.")
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = MSDUNet_MBV2_CorrCSA(num_classes=1).to(device)
    
    # Loss, Optimizer, Scheduler
    criterion = AdaptiveSegmentationLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Train
    train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        criterion=criterion, 
        device=device, 
        scheduler=scheduler, 
        num_epochs=args.epochs, 
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()
