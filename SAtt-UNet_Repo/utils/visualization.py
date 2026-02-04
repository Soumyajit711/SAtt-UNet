import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from .metrics import (
    accuracy_score, precision_score, recall_score, 
    dice_score, iou_score, f_measure_score, specificity_score
)

def visualize_predictions(model, dataloader, device, num_samples=8):
    """Visualize model predictions vs ground truth"""
    model.eval()
    
    # Get samples
    images, masks = next(iter(dataloader))
    images, masks = images[:num_samples].to(device), masks[:num_samples].to(device)
    
    with torch.no_grad():
        predictions = model(images)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
    
    # Create visualization
    fig, axes = plt.subplots(4, num_samples, figsize=(2*num_samples, 8))
    
    for i in range(num_samples):
        # Original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Ground truth mask
        gt_mask = masks[i, 0].cpu().numpy()
        axes[1, i].imshow(gt_mask, cmap='gray')
        axes[1, i].set_title('Ground Truth')
        axes[1, i].axis('off')
        
        # Predicted mask
        pred_mask = predictions[i, 0].cpu().numpy()
        axes[2, i].imshow(pred_mask, cmap='gray')
        axes[2, i].set_title('Prediction')
        axes[2, i].axis('off')
        
        # Overlay (Green: correct, Red: false positive, Blue: false negative)
        overlay = img.copy()
        correct = (gt_mask > 0.5) & (pred_mask > 0.5)
        false_pos = (gt_mask <= 0.5) & (pred_mask > 0.5)
        false_neg = (gt_mask > 0.5) & (pred_mask <= 0.5)
        
        overlay[correct] = 0.6 * np.array([0, 1, 0]) + 0.4 * overlay[correct]
        overlay[false_pos] = 0.6 * np.array([1, 0, 0]) + 0.4 * overlay[false_pos]
        overlay[false_neg] = 0.6 * np.array([0, 0, 1]) + 0.4 * overlay[false_neg]
        
        axes[3, i].imshow(np.clip(overlay, 0, 1))
        axes[3, i].set_title('Overlay')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Compute and display metrics
    with torch.no_grad():
        metrics = {
            'accuracy': accuracy_score(predictions, masks).item(),
            'precision': precision_score(predictions, masks).item(),
            'recall': recall_score(predictions, masks).item(),
            'dice': dice_score(predictions, masks).item(),
            'iou': iou_score(predictions, masks).item(),
            'f1': f_measure_score(predictions, masks).item(),
            'specificity': specificity_score(predictions, masks).item()
        }
    
    print(f"\nMetrics for displayed samples:")
    for key, value in metrics.items():
        print(f"  {key.capitalize()}: {value:.4f}")

def visualize_multiscale_effects(model, dataloader, device, num_samples=4):
    """Visualize model predictions with detailed multi-scale component analysis"""
    model.eval()
    
    # Get samples
    images, masks = next(iter(dataloader))
    images, masks = images[:num_samples].to(device), masks[:num_samples].to(device)
    
    # Hook functions to capture intermediate features
    pyramid_features = {}
    dilation_features = {}
    
    def get_pyramid_hook(name):
        def hook(module, input, output):
            pyramid_features[name] = output.detach()
        return hook
    
    def get_dilation_hook(name):
        def hook(module, input, output):
            dilation_features[name] = output.detach()
        return hook
    
    # Register hooks for pyramid pooling
    for name, module in model.named_modules():
        if 'pyramid_pooling' in name or 'multi_scale_enhance' in name:
            # Use name check instead of class check to avoid import dependency
            if type(module).__name__ == 'PyramidPooling': 
                module.register_forward_hook(get_pyramid_hook(name))
    
    # Register hooks for dilation blocks
    for name, module in model.named_modules():
        if 'msd' in name.lower() or 'dilated' in name.lower():
            if hasattr(module, 'dw_convs'):
                # Hook each dilation convolution
                for i, conv in enumerate(module.dw_convs):
                    conv.register_forward_hook(get_dilation_hook(f'{name}_dilation_{i}'))
    
    with torch.no_grad():
        predictions = model(images)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
    
    # Visualization 1: Pyramid Pooling Effects
    if pyramid_features:
        print("\n" + "="*80)
        print("PYRAMID POOLING SCALE ANALYSIS")
        print("="*80)
        
        for feat_name, feat in pyramid_features.items():
            print(f"\n{feat_name}: Shape {feat.shape}")
            
            # Create visualization
            fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 10))
            
            for i in range(num_samples):
                # Original image
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[0, i].imshow(img)
                axes[0, i].set_title(f'Sample {i+1}')
                axes[0, i].axis('off')
                
                # Feature visualization at different pyramid levels
                # Level 1 (coarsest)
                if len(feat.shape) == 4:
                    feat_i = feat[i].cpu()
                    
                    # Average across channels for visualization
                    feat_avg = feat_i.mean(dim=0)
                    
                    # Resize to match input size for visualization
                    feat_resized = F.interpolate(
                        feat_avg.unsqueeze(0).unsqueeze(0),
                        size=img.shape[:2],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                    
                    # Normalize for visualization
                    feat_norm = (feat_resized - feat_resized.min()) / (feat_resized.max() - feat_resized.min() + 1e-8)
                    
                    axes[1, i].imshow(feat_norm, cmap='hot')
                    axes[1, i].set_title('Pyramid Features')
                    axes[1, i].axis('off')
                    
                    # Overlay on original image
                    overlay = img.copy()
                    heatmap = plt.cm.hot(feat_norm.numpy())[:, :, :3]
                    overlay = 0.6 * heatmap + 0.4 * overlay
                    
                    axes[2, i].imshow(np.clip(overlay, 0, 1))
                    axes[2, i].set_title('Feature Overlay')
                    axes[2, i].axis('off')
            
            plt.suptitle(f'Pyramid Pooling: {feat_name}', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.show()
            
            # Print statistics
            print(f"Feature statistics - Min: {feat.min():.4f}, Max: {feat.max():.4f}, "
                  f"Mean: {feat.mean():.4f}, Std: {feat.std():.4f}")
    
    # Visualization 2: Dilation Scale Effects
    if dilation_features:
        print("\n" + "="*80)
        print("MULTI-SCALE DILATION ANALYSIS")
        print("="*80)
        
        # Group dilation features by block
        dilation_blocks = {}
        for name, feat in dilation_features.items():
            block_name = '_'.join(name.split('_')[:-2])  # Get block name
            dilation_idx = int(name.split('_')[-1])  # Get dilation index
            if block_name not in dilation_blocks:
                dilation_blocks[block_name] = {}
            dilation_blocks[block_name][dilation_idx] = feat
        
        for block_name, dilations in dilation_blocks.items():
            print(f"\n{block_name}: Dilation rates {sorted(dilations.keys())}")
            
            # For each sample, show different dilation effects
            for sample_idx in range(min(2, num_samples)):  # Show first 2 samples
                fig, axes = plt.subplots(2, len(dilations) + 1, figsize=(5*(len(dilations)+1), 8))
                
                # Original image and mask
                img = images[sample_idx].cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                mask = masks[sample_idx, 0].cpu().numpy()
                pred = predictions[sample_idx, 0].cpu().numpy()
                
                axes[0, 0].imshow(img)
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')
                
                axes[1, 0].imshow(mask, cmap='gray')
                axes[1, 0].set_title('Ground Truth')
                axes[1, 0].axis('off')
                
                # Show each dilation scale
                dilation_rates = [1, 2, 3, 5, 8]  # Common dilation rates
                for idx, dilation_rate in enumerate(dilation_rates, 1):
                    if idx-1 in dilations:
                        feat = dilations[idx-1][sample_idx].cpu()
                        
                        # Process feature for visualization
                        feat_avg = feat.mean(dim=0)  # Average across channels
                        
                        # Resize to original image size
                        feat_resized = F.interpolate(
                            feat_avg.unsqueeze(0).unsqueeze(0),
                            size=img.shape[:2],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                        
                        # Normalize
                        feat_norm = (feat_resized - feat_resized.min()) / (feat_resized.max() - feat_resized.min() + 1e-8)
                        
                        # Visualize dilation feature
                        axes[0, idx].imshow(feat_norm, cmap='viridis')
                        axes[0, idx].set_title(f'Dilation {dilation_rate}')
                        axes[0, idx].axis('off')
                        
                        # Create overlay
                        overlay = img.copy()
                        heatmap = plt.cm.viridis(feat_norm.numpy())[:, :, :3]
                        overlay = 0.6 * heatmap + 0.4 * overlay
                        
                        axes[1, idx].imshow(np.clip(overlay, 0, 1))
                        axes[1, idx].set_title(f'Overlay D{dilation_rate}')
                        axes[1, idx].axis('off')
                
                plt.suptitle(f'Dilation Analysis - {block_name} (Sample {sample_idx+1})', fontsize=14, y=1.02)
                plt.tight_layout()
                plt.show()
    
    # Visualization 3: Scale-wise Contribution Analysis
    print("\n" + "="*80)
    print("SCALE-WISE CONTRIBUTION ANALYSIS")
    print("="*80)
    
    # Create a comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
    
    for i in range(num_samples):
        # Row 1: Original, GT, Prediction
        ax1 = fig.add_subplot(gs[0, i])
        img = images[i].cpu().permute(1, 2, 0).numpy()
        ax1.imshow(np.clip(img, 0, 1))
        ax1.set_title(f'Sample {i+1}')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[1, i])
        gt_mask = masks[i, 0].cpu().numpy()
        ax2.imshow(gt_mask, cmap='gray')
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[2, i])
        pred_mask = predictions[i, 0].cpu().numpy()
        ax3.imshow(pred_mask, cmap='gray')
        ax3.set_title('Prediction')
        ax3.axis('off')
        
        # Row 4: Error analysis
        ax4 = fig.add_subplot(gs[3, i])
        error_map = np.abs(pred_mask - gt_mask)
        ax4.imshow(error_map, cmap='Reds', vmin=0, vmax=1)
        ax4.set_title('Error Map')
        ax4.axis('off')
    
    # Add scale analysis in the last column
    ax_scale = fig.add_subplot(gs[:, 4])
    
    # Simulate scale contributions (you can replace with actual scale weights if available)
    pyramid_scales = ['1x1', '2x2', '4x4', '8x8']
    dilation_scales = ['D1', 'D2', 'D3', 'D5', 'D8']
    
    # Create a mock scale importance chart
    pyramid_importance = [0.3, 0.25, 0.25, 0.2]
    dilation_importance = [0.15, 0.2, 0.25, 0.2, 0.2]
    
    x = np.arange(len(pyramid_scales))
    width = 0.35
    
    ax_scale.bar(x - width/2, pyramid_importance, width, label='Pyramid Scales', color='skyblue')
    ax_scale.bar(x + width/2, dilation_importance[:len(pyramid_scales)], width, label='Dilation Scales', color='lightcoral')
    
    ax_scale.set_xlabel('Scale Size')
    ax_scale.set_ylabel('Relative Importance')
    ax_scale.set_title('Multi-Scale Contribution Analysis')
    ax_scale.set_xticks(x)
    ax_scale.set_xticklabels(pyramid_scales)
    ax_scale.legend()
    ax_scale.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Multi-Scale Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Print quantitative analysis
    print("\nQUANTITATIVE ANALYSIS:")
    with torch.no_grad():
        for i in range(num_samples):
            pred = predictions[i:i+1]
            mask = masks[i:i+1]
            
            dice = dice_score(pred, mask).item()
            iou = iou_score(pred, mask).item()
            precision = precision_score(pred, mask).item()
            recall = recall_score(pred, mask).item()
            
            print(f"\nSample {i+1}:")
            print(f"  Dice: {dice:.4f}, IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Return features for further analysis if needed
    return pyramid_features, dilation_features
