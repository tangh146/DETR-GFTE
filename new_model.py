import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from new_data import create_dataloaders
from utils import get_psuedo_knn, get_feature_vec

class GraphNetwork(nn.Module):
    def __init__(self, d_model, num_classes):
        super(GraphNetwork, self).__init__()
        self.conv1 = GCNConv(8, d_model)
        self.conv2 = GCNConv(d_model, d_model)
        self.lin1 = torch.nn.Linear(d_model * 2, d_model)
        self.lin_final = torch.nn.Linear(d_model, num_classes)

    def forward(self, feature_vec, bboxes, bbox_indices):
    
        edge_index = get_psuedo_knn(bboxes).to(feature_vec.device)

        # GCN Layers
        x = self.conv1(feature_vec, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Pairing Node Features
        x1 = x[edge_index[0]]
        x2 = x[edge_index[1]]
        xpair = torch.cat((x1, x2), dim=-1)  # Shape: [num_edges, d_model * 2]
        xpair = F.relu(self.lin1(xpair))

        # Final Classification
        xfin = self.lin_final(xpair)
        probs = F.log_softmax(xfin, dim=-1)  # Shape: [num_edges, num_classes]

        bbox_pairs = torch.cat((
            bboxes[edge_index[0]],  # Start bounding boxes
            bboxes[edge_index[1]]   # End bounding boxes
        ), dim=-1)

        bbox_index_pairs = torch.stack((
            bbox_indices[edge_index[0]],  # Start bounding boxes
            bbox_indices[edge_index[1]]   # End bounding boxes
        ), dim=1)

        return probs, bbox_pairs, bbox_index_pairs

# LightningModule for training
class GNLightning(pl.LightningModule):
    def __init__(self, d_model, lr):
        super(GNLightning, self).__init__()
        self.CLASS_NO_REL, self.CLASS_HORZ_REL, self.CLASS_VERT_REL = 0, 1, 2
        self.gnet = GraphNetwork(d_model, num_classes=3)

        class_weights = torch.tensor([1.0, 1.5, 1.2], dtype=torch.float32)

        self.criterion = torch.nn.NLLLoss()
        self.lr = lr

        self.dataloaders = create_dataloaders(
            train_label_path=r'C:\Users\tangy\Downloads\DETR-GFTE\datasets\gnet_train.jsonl',
            val_label_path=r'C:\Users\tangy\Downloads\DETR-GFTE\datasets\gnet_val.jsonl')

    def forward(self, batch):
        return self.common_step(batch)
    
    def common_step(self, batch):

        table_grid = batch['table_grid']
        gt_bboxes = batch['gt_bboxes'] # shape (num_bboxes, 4)
        gt_bbox_indices = batch['gt_bbox_indices']
        orig_size = batch['orig_size'] # 2 items: width and height

        feature_vec = get_feature_vec(gt_bboxes, orig_size) # shape (num_bboxes, 8)

        probs, bbox_pairs, bbox_index_pairs = self.gnet(feature_vec, gt_bboxes, gt_bbox_indices)

        gt_classes = self.process_target(bbox_index_pairs, table_grid)

        return probs, gt_classes, bbox_pairs
        

    def training_step(self, batch, batch_idx):

        probs, gt_classes, _ = self.common_step(batch)

        loss = self.criterion(probs, gt_classes)

        loss_dict = {'loss': loss}

        # Debugging print statement
        for k, v in loss_dict.items():
            print(f"Batch {batch_idx} - {k}: {v.item()}")

        return loss

    def validation_step(self, batch, batch_idx):
        # hidden_state, pred_bboxes, labels = batch.x, batch.pos, batch.y
        # probs = self.model(hidden_state, pred_bboxes)
        # val_loss = self.criterion(probs, labels)
        # self.log('val_loss', val_loss)
        # return val_loss
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.dataloaders[0]

    def val_dataloader(self):
        return self.dataloaders[1]
    
    def process_target(self, bbox_index_pairs, table_grid):

        horz_map, vert_map = {}, {}

        for row_idx, row in enumerate(table_grid):
            for col_idx, item in enumerate(row):
                for tgt_idx in range(col_idx, len(row)):
                    if item != row[tgt_idx]:
                        horz_map[(int(item), int(row[tgt_idx]))] = None
                for tgt_idx in range(row_idx, len(table_grid)):
                    if item != table_grid[tgt_idx][col_idx]:
                        vert_map[(int(item), int(table_grid[tgt_idx][col_idx]))] = None

        gt_classes = []
        for pair in bbox_index_pairs:
            pair = tuple(map(int, pair.tolist()))
            reverse_pair = (pair[1], pair[0])
            # this commented code doesn't work!!!
            if pair in horz_map or reverse_pair in horz_map:
                gt_classes.append(self.CLASS_HORZ_REL)
            elif pair in vert_map or reverse_pair in vert_map:
                gt_classes.append(self.CLASS_VERT_REL)
            else:
                gt_classes.append(self.CLASS_NO_REL)
        
        return torch.tensor(gt_classes, dtype=torch.long).to(self.device)