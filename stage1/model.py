import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pytorch_lightning as pl
from data import create_dataloaders

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from utils import construct_basic_knn_graph, get_feature_vec

class GraphNetwork(nn.Module):
    def __init__(self, d_model, num_classes):
        super(GraphNetwork, self).__init__()
        self.conv1 = GCNConv(8, d_model)
        self.conv2 = GCNConv(d_model, d_model)
        self.lin1 = torch.nn.Linear(d_model * 2, d_model)
        self.lin_final = torch.nn.Linear(d_model, num_classes)

    def forward(self, data):

        feature_vec, edge_index = data.x, data.edge_index

        # GCN Layers
        x = self.conv1(feature_vec, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Pairing Node Features
        x1 = x[edge_index[0]]
        x2 = x[edge_index[1]]
        xpair = torch.cat((x1, x2), dim=1)  # Shape: [num_edges, d_model * 2]
        xpair = F.relu(self.lin1(xpair))

        # Final Classification
        xfin = self.lin_final(xpair)
        probs = F.log_softmax(xfin, dim=1)  # Shape: [num_edges, num_classes]

        return probs

# LightningModule for training
class GNLightning(pl.LightningModule):
    def __init__(self, d_model, lr, batch_size, num_workers, train_path, val_path):
        super(GNLightning, self).__init__()
        self.gnet = GraphNetwork(d_model, num_classes=3)
        self.criterion = torch.nn.NLLLoss()
        self.lr = lr
        self.dataloaders = create_dataloaders(
            train_label_path=train_path,
            val_label_path=val_path,
            batch_size=batch_size, num_workers=num_workers)

    def forward(self, bboxes, orig_size):
        # bboxes tensor of shape (num_bboxes, 4) in coco format
        # orig size tensor of shape (2)

        feature_vec = get_feature_vec(bboxes, orig_size)

        edge_index = construct_basic_knn_graph(bboxes)

        data = Data(x=feature_vec, edge_index=edge_index).to(self.device)

        probs = self.gnet(data)

        return probs, edge_index
    
    def common_step(self, batch):

        data, gt_classes = batch
        data, gt_classes = data.to(self.device), gt_classes.to(self.device)

        probs = self.gnet(data)

        gt_classes_flat = gt_classes.view(-1)

        valid_gt_classes = gt_classes_flat[gt_classes_flat != -1]

        loss = self.criterion(probs, valid_gt_classes)

        # Compute accuracy
        _, predicted_classes = probs.max(dim=1)  # Get predicted class indices
        correct_predictions = (predicted_classes == valid_gt_classes).sum().item()
        total_predictions = valid_gt_classes.size(0)  # Count of valid predictions
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Include accuracy in loss_dict
        loss_dict = {'loss': loss, "accuracy":torch.tensor(accuracy, device=self.device)}

        return loss,accuracy, loss_dict
        

    def training_step(self, batch, batch_idx):

        try:
            loss,accuracy, loss_dict = self.common_step(batch)
        except Exception as e:
            print(f"Error: {e}")
            return None

        # Debugging print statement
        for k, v in loss_dict.items():
            print(f"Batch {batch_idx} - {k}: {v.item()}")

        self.log("training_loss", loss)
        self.log("training_accuracy",accuracy)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

         # Save checkpoint halfway through the epoch
        if batch_idx % 10 == 0:  # Example condition
            self.trainer.save_checkpoint(f"checkpoint-step-{self.current_epoch}-{batch_idx}.ckpt")
        

        return loss

    def validation_step(self, batch, batch_idx):

        try:
            loss,accuracy, loss_dict = self.common_step(batch)
        except Exception as e:
            print(f"Error: {e}")
            return None
            
        # Debugging step to confirm `validation_loss` is logged
        print(f"Logging validation_loss: {loss}")
        print(f"Logging validation_accuracy: {accuracy}")
        
        self.log("validation_loss", loss)
        self.log("validation_accuracy", accuracy)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.dataloaders[0]

    def val_dataloader(self):
        return self.dataloaders[1]