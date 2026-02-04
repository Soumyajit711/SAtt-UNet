import torch
from tqdm import tqdm
from utils.metrics import (
    accuracy_score, precision_score, recall_score, 
    dice_score, iou_score, f_measure_score, specificity_score
)
from utils.visualization import visualize_predictions, visualize_multiscale_effects

def train_model(model, train_loader, val_loader, optimizer, criterion, device, scheduler=None, num_epochs=100, save_path='best_model.pth'):
    best_val_iou = 0.0  # track best IoU
    train_losses, val_losses = [], []
    train_metrics = {m: [] for m in ['accuracy','precision','recall','dice','iou','fmeasure','sensitivity','specificity']}
    val_metrics   = {m: [] for m in ['accuracy','precision','recall','dice','iou','fmeasure','sensitivity','specificity']}
    
    for epoch in range(num_epochs):
        # ---------------------- Training ----------------------
        model.train()
        epoch_train_loss = 0.0
        epoch_train_metrics = {m:0.0 for m in train_metrics}
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images, return_aux=True)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            with torch.no_grad():
                pred = outputs[0] if isinstance(outputs, tuple) else outputs
                epoch_train_metrics['accuracy']     += accuracy_score(pred, masks).item()
                epoch_train_metrics['precision']    += precision_score(pred, masks).item()
                epoch_train_metrics['recall']       += recall_score(pred, masks).item()
                epoch_train_metrics['dice']         += dice_score(pred, masks).item()
                epoch_train_metrics['iou']          += iou_score(pred, masks).item()
                epoch_train_metrics['fmeasure']     += f_measure_score(pred, masks).item()
                epoch_train_metrics['sensitivity']  += recall_score(pred, masks).item()
                epoch_train_metrics['specificity']  += specificity_score(pred, masks).item()
        
        # average
        train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(train_loss)
        for m in train_metrics: train_metrics[m].append(epoch_train_metrics[m] / len(train_loader))
        
        # ---------------------- Validation ----------------------
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_metrics = {m:0.0 for m in val_metrics}
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item()
                
                epoch_val_metrics['accuracy']     += accuracy_score(outputs, masks).item()
                epoch_val_metrics['precision']    += precision_score(outputs, masks).item()
                epoch_val_metrics['recall']       += recall_score(outputs, masks).item()
                epoch_val_metrics['dice']         += dice_score(outputs, masks).item()
                epoch_val_metrics['iou']          += iou_score(outputs, masks).item()
                epoch_val_metrics['fmeasure']     += f_measure_score(outputs, masks).item()
                epoch_val_metrics['sensitivity']  += recall_score(outputs, masks).item()
                epoch_val_metrics['specificity']  += specificity_score(outputs, masks).item()
        
        val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(val_loss)
        for m in val_metrics: val_metrics[m].append(epoch_val_metrics[m] / len(val_loader))
        
        if scheduler: scheduler.step()
        
        # ---------------------- Save Best ----------------------
        current_val_iou = val_metrics['iou'][-1]
        if current_val_iou > best_val_iou:
            best_val_iou = current_val_iou
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model updated at Epoch {epoch+1} | IoU: {current_val_iou:.4f}")
        
        # ---------------------- Logs ----------------------
        print(f'\nEpoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print('\nTraining Metrics:') 
        print(f'Accuracy: {train_metrics["accuracy"][-1]:.4f}, Precision: {train_metrics["precision"][-1]:.4f}, Recall: {train_metrics["recall"][-1]:.4f}, Dice: {train_metrics["dice"][-1]:.4f}, IoU: {train_metrics["iou"][-1]:.4f}, F-Measure: {train_metrics["fmeasure"][-1]:.4f}, Sensitivity: {train_metrics["sensitivity"][-1]:.4f}, Specificity: {train_metrics["specificity"][-1]:.4f}') 
        print('\nValidation Metrics:') 
        print(f'Accuracy: {val_metrics["accuracy"][-1]:.4f}, Precision: {val_metrics["precision"][-1]:.4f}, Recall: {val_metrics["recall"][-1]:.4f}, Dice: {val_metrics["dice"][-1]:.4f}, IoU: {val_metrics["iou"][-1]:.4f}, F-Measure: {val_metrics["fmeasure"][-1]:.4f}, Sensitivity: {val_metrics["sensitivity"][-1]:.4f}, Specificity: {val_metrics["specificity"][-1]:.4f}')
        
        # Show visualizations every N epochs
        if (epoch + 1) % 50 == 0:
            print(f"\nGenerating visualizations for epoch {epoch+1}...")
            visualize_predictions(model, val_loader, device, num_samples=5)

            print(f"\nGenerating detailed multi-scale visualizations for epoch {epoch+1}...")
            # Note: visualize_multiscale_effects prints its own output
            visualize_multiscale_effects(model, val_loader, device, num_samples=5)
    
    return train_losses, val_losses, train_metrics, val_metrics
