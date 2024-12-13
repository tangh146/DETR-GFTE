import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms as T
from torchvision.ops import box_iou
from PIL import Image
import json
from functools import lru_cache
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import numpy as np
import datetime
import math
import time

# Distributed Setup
def setup_distributed():
    """Initialize the distributed environment."""
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))



def convert_to_xyxy(boxes):
    """
    Detect and convert bounding boxes from xywh to xyxy format if necessary.

    Args:
        boxes (Tensor, list, or single box): Bounding boxes in either xywh or xyxy format.
                                             - xywh: [x, y, w, h]
                                             - xyxy: [x1, y1, x2, y2]

    Returns:
        Tensor: Bounding boxes in xyxy format.
    """
    # If input is a single box (list or tuple), wrap it in a tensor
    if isinstance(boxes, (list, tuple)):
        boxes = torch.tensor([boxes], dtype=torch.float32)

    # Ensure boxes are in tensor format
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)

    # Detect format: Check if boxes are in xywh
    # Condition: x2 < x1 or y2 < y1 implies xywh, as w and h cannot be negative.
    if (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all():
        # Boxes are already in xyxy format
        return boxes

    # Convert xywh to xyxy
    x, y, w, h = boxes.unbind(1)  # Split into components
    x2 = x + w  # Bottom-right x-coordinate
    y2 = y + h  # Bottom-right y-coordinate
    boxes_xyxy = torch.stack((x, y, x2, y2), dim=1)

    return boxes_xyxy


# Dataset Definition
default_transform = T.Compose([T.ToTensor()])
class PubTabNetDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_file, split="train", transforms=None, limit_samples=False):
        """
        Args:
            images_dir (str): Directory containing images.
            annotations_file (str): Path to the annotations file.
            split (str): Dataset split ("train" or "val").
            transforms (callable, optional): Transformations to apply to images.
            limit_samples (bool): If True, use only the first 100 samples.
        """
        self.images_dir = images_dir
        self.transforms = transforms if transforms else default_transform
        self.annotations_file = annotations_file
        self.split = split
        self.line_offsets = self._get_line_offsets(limit_samples)
        print(f"Total valid samples in {split} dataset: {len(self.line_offsets)}")

    def _get_line_offsets(self, limit_samples):
        """
        Get the line offsets of valid samples from the annotations file.
        
        Args:
            limit_samples (bool): If True, limit the samples to the first 100.

        Returns:
            list: Line offsets of valid samples.
        """
        offsets = []
        with open(self.annotations_file, 'r', encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                obj = json.loads(line)
                if (obj.get('split') == self.split and "html" in obj and "cells" in obj["html"] and self.image_exists(obj['filename'])):
                    offsets.append(offset)
                # Stop collecting after 100 valid samples if limit_samples is True
                if limit_samples and len(offsets) >= 500:
                    break
        return offsets

    def image_exists(self, filename):
        """Check if an image file exists."""
        return os.path.exists(os.path.join(self.images_dir, filename))

    def load_image(self, filename):
        """Load an image from the specified filename."""
        image_path = os.path.join(self.images_dir, filename)
        try:
            return Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: {filename} not found in {self.images_dir}. Skipping...")
            return None

    def __len__(self):
        """Return the total number of samples."""
        return len(self.line_offsets)

    def __getitem__(self, idx):
        """Retrieve an image and its corresponding annotation."""
        offset = self.line_offsets[idx]
        with open(self.annotations_file, 'r', encoding="utf-8") as f:
            f.seek(offset)
            annotation = json.loads(f.readline())

        image = self.load_image(annotation['filename'])
        if image is None:  # Skip if the image is missing
            return None

        boxes = [cell['bbox'] for cell in annotation['html']['cells'] if 'bbox' in cell]
        
        if not boxes:  # If no bounding boxes exist
            return None

        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes = convert_to_xyxy(boxes) 
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        if self.transforms:
            image = self.transforms(image)

        return image, {"boxes": boxes, "labels": labels}



# Collate Function
def valid_collate(batch):
    return tuple(zip(*[b for b in batch if b[0] is not None]))

# Model
def get_model():
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    return FasterRCNN(backbone, num_classes=2)
def save_checkpoint(model, optimizer, epoch, checkpoint_dir="checkpoints", scheduler=None):
    """
    Save model and optimizer state for resuming training.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer whose state to save.
        epoch (int): Current epoch number.
        checkpoint_dir (str): Directory to save checkpoints.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler state.
    """
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the checkpoint file path
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

    # Save the model, optimizer, and scheduler state (if provided)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)

    if dist.get_rank() == 0:  # Only rank 0 process saves checkpoint in DDP
        print(f"Checkpoint saved at: {checkpoint_path}")
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load model, optimizer, and scheduler states.

    Args:
        checkpoint_path (str): Path to the saved checkpoint.
        model (torch.nn.Module): Model to load the weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state.

    Returns:
        int: The epoch number to resume training.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cuda")

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded. Resuming from epoch {epoch}.")
    return epoch






def compute_iou(pred_boxes, gt_boxes):
    """
    Compute Intersection over Union (IoU) between predicted and ground truth boxes.
    
    Args:
        pred_boxes (Tensor): Predicted bounding boxes of shape (N, 4).
        gt_boxes (Tensor): Ground truth bounding boxes of shape (M, 4).

    Returns:
        Tensor: IoU matrix of shape (N, M).
    """
    pred_boxes = convert_to_xyxy(pred_boxes)
    gt_boxes = convert_to_xyxy(gt_boxes)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return torch.zeros((len(pred_boxes), len(gt_boxes)))  # Return zero IoU if no boxes
    
    iou = box_iou(pred_boxes, gt_boxes)  # Computes IoU for all box pairs
    return iou
    
def compute_precision_recall(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5, score_threshold=0.5):
    """
    Compute precision, recall, and F1 score for a single batch.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes of shape (N, 4).
        pred_scores (Tensor): Predicted confidence scores of shape (N,).
        gt_boxes (Tensor): Ground truth bounding boxes of shape (M, 4).
        iou_threshold (float): IoU threshold to determine true positives.
        score_threshold (float): Minimum confidence score to consider predictions.

    Returns:
        dict: Precision, recall, and F1 score.
    """
    pred_boxes = convert_to_xyxy(pred_boxes)
    gt_boxes = convert_to_xyxy(gt_boxes)
    # Filter boxes by confidence score
    keep = pred_scores >= score_threshold
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]

    # Compute IoU
    iou = compute_iou(pred_boxes, gt_boxes)

    # Initialize counters
    num_gt_boxes = gt_boxes.shape[0]
    tp = 0
    matched_gt = set()

    # Match predictions to ground truth
    for i in range(iou.shape[0]):  # For each predicted box
        max_iou, gt_idx = torch.max(iou[i], dim=0)
        if max_iou >= iou_threshold and gt_idx.item() not in matched_gt:
            tp += 1
            matched_gt.add(gt_idx.item())

    fp = len(pred_boxes) - tp  # False positives
    fn = num_gt_boxes - len(matched_gt)  # False negatives

    # Precision, Recall, and F1 Score
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {"precision": precision, "recall": recall, "f1_score": f1_score}
    
def compute_map(pred_boxes, pred_scores, gt_boxes, iou_thresholds=[0.5, 0.75, 0.95]):
    """
    Compute Mean Average Precision (mAP) over specified IoU thresholds.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes of shape (N, 4).
        pred_scores (Tensor): Predicted confidence scores of shape (N,).
        gt_boxes (Tensor): Ground truth bounding boxes of shape (M, 4).
        iou_thresholds (list): List of IoU thresholds to compute AP.

    Returns:
        float: Mean Average Precision (mAP).
    """
    pred_boxes = convert_to_xyxy(pred_boxes)
    gt_boxes = convert_to_xyxy(gt_boxes)
    aps = []
    for iou_thresh in iou_thresholds:
        metrics = compute_precision_recall(pred_boxes, pred_scores, gt_boxes, iou_threshold=iou_thresh)
        aps.append(metrics["precision"])  # AP is the precision at IoU threshold

    map_value = sum(aps) / len(iou_thresholds)  # Mean over all thresholds
    return map_value

def train_one_epoch(model, train_loader, optimizer, scaler, scheduler, device, epoch, accumulation_steps=1, print_freq=10):
    model.train()
    train_loader.sampler.set_epoch(epoch)
    train_batch_metrics_ls = []
    epoch_loss = 0.0
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values()) / accumulation_steps

            scaler.scale(total_loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            epoch_loss += total_loss.item()

            if dist.get_rank() == 0 and (batch_idx + 1) % 55 == 0:
                # Compute metrics
                model.eval()
                with torch.no_grad():
                    iou_list, precision_list, recall_list, f1_score_list, mAP_list = [], [], [], [], []
                    outputs = model(images)
                    for output, target in zip(outputs, targets):
                        pred_boxes = convert_to_xyxy(output.get("boxes", torch.tensor([])).detach().cpu())
                        pred_scores = output.get("scores", torch.tensor([])).detach().cpu()
                        gt_boxes = convert_to_xyxy(target["boxes"].detach().cpu())

                        iou = compute_iou(pred_boxes, gt_boxes)
                        pr_metrics = compute_precision_recall(pred_boxes, pred_scores, gt_boxes)
                        mAP = compute_map(pred_boxes, pred_scores, gt_boxes)

                        iou_list.append(iou.mean().item())
                        precision_list.append(pr_metrics["precision"])
                        recall_list.append(pr_metrics["recall"])
                        f1_score_list.append(pr_metrics["f1_score"])
                        mAP_list.append(mAP)

                # Append metrics (only on rank 0)
               
                train_batch_metrics_ls.append({
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                        "loss_classifier": loss_dict.get("loss_classifier", torch.tensor(float('nan'))).item(),
                        "loss_box_reg": loss_dict.get("loss_box_reg", torch.tensor(float('nan'))).item(),
                        "loss_objectness": loss_dict.get("loss_objectness", torch.tensor(float('nan'))).item(),
                        "loss_rpn_box_reg": loss_dict.get("loss_rpn_box_reg", torch.tensor(float('nan'))).item(),
                        "total_loss": total_loss.item(),
                        "iou": np.nanmean(iou_list),
                        "precision": np.nanmean(precision_list),
                        "recall": np.nanmean(recall_list),
                        "f1_score": np.nanmean(f1_score_list),
                        "mAP": np.nanmean(mAP_list),
                    })
                model.train()

                # Print progress (only on rank 0)
                if dist.get_rank() == 0 and (batch_idx + 1) % print_freq == 0:
                    print(f"[Batch {batch_idx+1}] Loss: {total_loss.item():.4f}")

        except Exception as e:
            print(f"Warning: Batch {batch_idx+1} failed with error: {e}")

    if dist.get_rank() == 0:
        avg_loss = epoch_loss / len(train_loader)
        end_time = time.time()
        print(f"Training Epoch {epoch+1} Completed. Average Loss: {avg_loss:.4f}. Time: {end_time - start_time:.2f} seconds.")

    return train_batch_metrics_ls if dist.get_rank() == 0 else None


def validate_one_epoch(model, val_loader, device, epoch, print_freq=10):
    model.eval()  # Explicitly set model to evaluation mode

    val_batch_metrics_ls=[]
    

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")):
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Compute losses
                with torch.amp.autocast(device_type='cuda'):
                    model.train()  # Temporarily switch to train mode for loss computation
                    loss_dict = model(images, targets)
                    model.eval()

                total_loss = sum(loss for loss in loss_dict.values())

                # Compute predictions and batch-averaged metrics
                outputs = model(images)
                iou_list = []
                precision_list = []
                recall_list = []
                f1_score_list = []
                mAP_list = []
                print("num outputs in validation : ",len(outputs))
                for output, target in zip(outputs, targets):

                        pred_boxes = convert_to_xyxy(output.get("boxes", torch.tensor([])).detach().cpu())
                        pred_scores = output.get("scores", torch.tensor([])).detach().cpu()
                        gt_boxes = convert_to_xyxy(target["boxes"].detach().cpu())
    
                        
                        iou = compute_iou(pred_boxes, gt_boxes) 
                        pr_metrics = compute_precision_recall(pred_boxes, pred_scores, gt_boxes) 
                        mAP = compute_map(pred_boxes, pred_scores, gt_boxes)
    
                        #iou_list.append(iou.mean().item() if iou.numel() > 0 else float('nan'))
                        iou_list.append(iou.mean().item() )
                        precision_list.append(pr_metrics["precision"])
                        recall_list.append(pr_metrics["recall"])
                        f1_score_list.append(pr_metrics["f1_score"])
                        mAP_list.append(mAP)

                # Append averaged metrics for the entire batch
                
                val_batch_metrics = {
                        "epoch":(epoch+1),
                        "batch":(batch_idx+1),
                        "loss_classifier": loss_dict.get("loss_classifier", torch.tensor(float('nan'))).item(),
                        "loss_box_reg": loss_dict.get("loss_box_reg", torch.tensor(float('nan'))).item(),
                        "loss_objectness": loss_dict.get("loss_objectness", torch.tensor(float('nan'))).item(),
                        "loss_rpn_box_reg": loss_dict.get("loss_rpn_box_reg", torch.tensor(float('nan'))).item(),
                        "total_loss": total_loss.item(),
                        "iou": np.nanmean(iou_list),
                        "precision": np.nanmean(precision_list),
                        "recall": np.nanmean(recall_list),
                        "f1_score": np.nanmean(f1_score_list),
                        "mAP": np.nanmean(mAP_list),
                    }
                val_batch_metrics_ls.append(val_batch_metrics)

            except Exception as e:
                print(f"Warning: Batch {batch_idx+1} failed with error: {e}")
                # Append NaN for all metrics in case of error

                val_batch_metrics = {
                    "epoch":(epoch+1),
                    "batch":(batch_idx+1),
                    "loss_classifier": float('nan'),
                    "loss_box_reg": float('nan'),
                    "loss_objectness": float('nan'),
                    "loss_rpn_box_reg": float('nan'),
                    "total_loss": float('nan'),
                    "iou": float('nan'),
                    "precision":float('nan'),
                    "recall": float('nan'),
                    "f1_score": float('nan'),
                    "mAP":float('nan'),
                }
                val_batch_metrics_ls.append(val_batch_metrics)

            # Print progress
            if (batch_idx + 1) % print_freq == 0:
                print(f"[Batch {batch_idx+1}/{len(val_loader)}] "
                      f"Loss: {val_batch_metrics['total_loss']:.4f}, "
                      f"IoU: {val_batch_metrics['iou']:.4f}, "
                      f"Precision: {val_batch_metrics['precision']:.4f}, "
                      f"Recall: {val_batch_metrics['recall']:.4f}, "
                      f"F1: {val_batch_metrics['f1_score']:.4f}, "
                      f"mAP: {val_batch_metrics['mAP']:.4f}")

    print(f"Validation Epoch {epoch+1} Completed.")
    return val_batch_metrics_ls



    
def save_metrics_to_json(epochs_metrics, file_path, phase="train"):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    data[phase] = []
    for metrics in epochs_metrics:
        sanitized_metrics = {}      
        for k, v in metrics.items():
            if isinstance(v, float) and math.isnan(v):
                # Store NaN as None for JSON compatibility
                sanitized_metrics[k] = None
            else:
                sanitized_metrics[k] = v
            
        data[phase].append(sanitized_metrics)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def plot_metrics(train_metrics, val_metrics, metric_name, save_path=None):
    """
    Plot training and validation metrics over epochs.

    Args:
        train_metrics (dict): Training metrics dictionary.
        val_metrics (dict): Validation metrics dictionary.
        metric_name (str): Name of the metric to plot (e.g., 'total_loss', 'mAP', 'iou').
        save_path (str, optional): Path to save the plot as an image. Default is None.
    """
    # Extract the metric values
    train_values = [sum(train_metrics[metric_name]) / len(train_metrics[metric_name]) for _ in range(len(train_metrics["total_loss"]))]
    val_values = [sum(val_metrics[metric_name]) / len(val_metrics[metric_name]) for _ in range(len(val_metrics["total_loss"]))]
    
    epochs = range(1, len(train_values) + 1)

    # Plot the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, label=f"Training {metric_name}", marker='o')
    plt.plot(epochs, val_values, label=f"Validation {metric_name}", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(f"Training vs Validation {metric_name.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
# Training Script
def main():
    setup_distributed()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    rank = int(os.environ["RANK"])

    # Dataset and DataLoader
    train_dataset = PubTabNetDataset(
        images_dir="pubtabnet/train", annotations_file="pubtabnet/PubTabNet_2.0.0.jsonl", split="train",limit_samples=False
    )
    val_dataset = PubTabNetDataset(
        images_dir="pubtabnet/val", annotations_file="pubtabnet/PubTabNet_2.0.0.jsonl", split="val",limit_samples=False
    )
    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)

    print(f"Total training samples: {num_train_samples}")
    print(f"Total validation samples: {num_val_samples}")

    train_loader = DataLoader(
        train_dataset, batch_size=32, sampler=DistributedSampler(train_dataset), collate_fn=valid_collate,num_workers=16, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, sampler=DistributedSampler(val_dataset), collate_fn=valid_collate,num_workers=16,pin_memory=True
    )

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    
    print(f"Total training batches: {num_train_batches}")
    print(f"Total validation batches: {num_val_batches}")

    

    # Model
    model = get_model().to(device)
    model = DDP(model, device_ids=[device])
    # # Freeze backbone
    # for param in model.module.backbone.parameters():
    #     param.requires_grad = False

    # Training Loop
    num_epochs = 15
    accumulation_steps = 4
    trial="full 15 optimized lr"
    
    # Optimizer and Scheduler
    
    
    
    
    total_steps = len(train_loader) * num_epochs
    
    base_lr = 1e-4
    lr = base_lr * 8  # Scale with 8 GPUs
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1)

    
    scaler = GradScaler()

   
    # Training Loop
    best_map = 0.0  # Track the best mAP
    patience = 5    # Number of epochs to wait for improvement
    counter = 0     # Counter for epochs without improvement

    train_metrics_epochs=[]
    val_metrics_epochs=[]
    
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
    
        # 1. Train for one epoch
        train_one_epoch_metrics = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, device, epoch,print_freq=10)
    
        # 2. Validate after training
        val_one_epoch_metrics = validate_one_epoch(model, val_loader, device, epoch,print_freq=10)
        
    
        # 3. Compute and log average mAP for the epoch
        mAP_values_in_val = [d["mAP"] for d in val_one_epoch_metrics]
    
        avg_mAP = sum(mAP_values_in_val) / len(mAP_values_in_val)
        print(f"Epoch {epoch+1} Average Validation mAP: {avg_mAP:.4f}")
    
        # 4. Save checkpoint at the end of each epoch
        if dist.get_rank() == 0:
            print(f"Saving checkpoint for Epoch {epoch+1}...")
            save_checkpoint(model, optimizer, epoch, checkpoint_dir=trial, scheduler=scheduler)
    
        # 5. Check for early stopping and save the best model
        if avg_mAP > best_map:
            print(f"New best mAP: {avg_mAP:.4f}. Saving best model...")
            if dist.get_rank() == 0:
                save_checkpoint(model, optimizer, epoch, checkpoint_dir=trial, scheduler=scheduler)
            best_map = avg_mAP
            counter = 0  # Reset counter when improvement is found
        else:
            counter += 1  # Increment counter if no improvement
            print(f"No improvement in mAP. Early stopping counter: {counter}/{patience}")
    
        # 6. Early stopping condition
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best mAP: {best_map:.4f}")
            break
    
        # 7. Save metrics to Excel
        if dist.get_rank() == 0:
            
            
            filename= "metrics_log_asOfEpoch"+str(epoch+1)+"_"+trial+".json"
            train_metrics_epochs.extend(train_one_epoch_metrics)
            val_metrics_epochs.extend(val_one_epoch_metrics)
            save_metrics_to_json(train_metrics_epochs, filename, phase="train")
            save_metrics_to_json(val_metrics_epochs, filename, phase="val")
        
       
        
    
    print("Training Complete!")
    dist.destroy_process_group()


     
if __name__ == "__main__":
    main()
