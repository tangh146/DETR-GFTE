import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import create_dataloaders
from utils import load_config
import sys
import os
from datetime import datetime
from models import ViTWithMLPClassifier

def main(config_path):
    config = load_config(config_path)
    
    print("Starting train...")

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        config['training']['train_data_path'],
        config['training']['train_labels_path'],
        config['training']['val_data_path'],
        config['training']['val_labels_path'],
        config['training']['batch_size']
    )

    lr = config['training']['lr']
    num_epochs = config['training']['num_epochs']
    batches_before_val = config['training']['batches_before_val']
    checkpoint_save_dir = config['training']['checkpoint_save_dir']
    pretrained_model_path = config['training']['pretrained_model']
    
    device = torch.device('cuda:0')

    # Initialize the model, loss function, and optimizer
    model = ViTWithMLPClassifier().to(device)
    criterion = nn.CrossEntropyLoss()  # No ignoring of -1 class labels
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load pretrained model if specified
    if pretrained_model_path:
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch in enumerate(train_loader):
            image_tensor = batch['image_tensor'].to(device).requires_grad_()
            gt_patch_indices = batch['gt_patch_indices'].to(device).long()  # Ensure gt_patch_indices is of type Long
            
            optimizer.zero_grad()

            logits = model(image_tensor)
            
            # Compute loss
            # logits: (batch_size, num_patches, num_classes)
            # gt_patch_indices: (batch_size, num_patches)
            batch_size, num_patches, num_classes = logits.shape
            logits = logits.view(batch_size * num_patches, num_classes)  # shape: (batch_size * num_patches, num_classes)
            gt_patch_indices = gt_patch_indices.view(batch_size * num_patches)  # shape: (batch_size * num_patches)

            loss = criterion(logits, gt_patch_indices)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == gt_patch_indices).sum().item()
            total_predictions += gt_patch_indices.size(0)

            # Print batch statistics
            accuracy = 100 * correct_predictions / total_predictions
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Batch {batch_idx+1}, Loss: {loss.item()}, Accuracy: {accuracy:.2f}%")

            # Validate and save checkpoint
            if (batch_idx + 1) % batches_before_val == 0:
                print("Validating...")
                val_loss, val_accuracy = validate(model, val_loader, criterion, device)
                print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy:.2f}%")
                
                # Save checkpoint
                checkpoint_filename = datetime.now().strftime('%Y%m%d_%H%M%S') + ".pt"
                checkpoint_path = os.path.join(checkpoint_save_dir, checkpoint_filename)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in val_loader:
            image_tensor = batch['image_tensor'].to(device)
            gt_patch_indices = batch['gt_patch_indices'].to(device).long()  # Ensure gt_patch_indices is of type Long

            logits = model(image_tensor)
            batch_size, num_patches, num_classes = logits.shape
            logits = logits.view(batch_size * num_patches, num_classes)
            gt_patch_indices = gt_patch_indices.view(batch_size * num_patches)

            loss = criterion(logits, gt_patch_indices)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == gt_patch_indices).sum().item()
            total_predictions += gt_patch_indices.size(0)

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct_predictions / total_predictions
    return val_loss, val_accuracy

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <path_to_config_file>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)