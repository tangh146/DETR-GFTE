import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from data import create_dataloaders
from utils import state_dict_cleaner

class SimplifiedTbNet(nn.Module):
    def __init__(self, d_model, num_classes, num_nodes):
        super(SimplifiedTbNet, self).__init__()
        self.conv1 = GCNConv(d_model, d_model)
        self.conv2 = GCNConv(d_model, d_model)
        self.lin1 = torch.nn.Linear(d_model * 2, d_model)
        self.lin_final = torch.nn.Linear(d_model, num_classes)
        
        # create an edge index row = 2, col = num_edges
            # 0
            # 1
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).T
        # add the vertically flipped version of itself horizontally, to make the graph undirected
            # 0 1
            # 1 0
        self.edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        self.num_nodes = num_nodes

    def forward(self, hidden_state, pred_bboxes):

        # shape is verified!
        batch_size, num_nodes, d_model = hidden_state.shape

        # Flatten the batch dimension for GCNConv
        # (batch_size, num_nodes, d_model) -> (batch_size * num_nodes, d_model) verified!
        hidden_state_reshaped = hidden_state.view(-1, d_model)

        # Repeat for each graph in the batch, horizontally. let's say batch size = 2
            # 0 1 0 1
            # 1 0 1 0
        edge_index_batch = self.edge_index.repeat(1, batch_size)
        # distinguish each graph in the batch
            # 0 1 2 3
            # 1 0 3 2  
        batch_offsets = torch.arange(batch_size).repeat_interleave(self.edge_index.size(1)) * self.num_nodes
        edge_index_batch += batch_offsets

        edge_index_batch = edge_index_batch.to(hidden_state_reshaped.device)

        # GCN Layers
        x = self.conv1(hidden_state_reshaped, edge_index_batch)
        x = F.relu(x)
        x = self.conv2(x, edge_index_batch)
        x = F.relu(x)

        # Reshape to separate batch dimension again
        x = x.view(batch_size, num_nodes, d_model)

        # Pairing Node Features
        x1 = x[:, self.edge_index[0]]
        x2 = x[:, self.edge_index[1]]
        xpair = torch.cat((x1, x2), dim=-1)  # Shape: [batch_size, num_edges, d_model * 2]
        xpair = F.relu(self.lin1(xpair))

        # Final Classification
        xfin = self.lin_final(xpair)
        probs = F.log_softmax(xfin, dim=-1)  # Shape: [batch_size, num_edges, num_classes]

        # Process Bounding Boxes
        bbox_pairs = torch.cat((
            pred_bboxes[:, self.edge_index[0]],  # Start bounding boxes
            pred_bboxes[:, self.edge_index[1]]   # End bounding boxes
        ), dim=-1)

        # bbox pairs shape = [batch size, num edges, bbox dim * 2]
        # probs shape = [batch size, num edges, num classes]
        return probs, bbox_pairs

class Detr(pl.LightningModule):

    def __init__(
            self, 
            train, 
            lr, 
            lr_backbone, 
            weight_decay, 
            pretrained, 
            checkpoint, 
            config
            ):
        
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=pretrained, 
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        if checkpoint:
            state_dict = torch.load(checkpoint)['state_dict']
            state_dict = state_dict_cleaner(state_dict)
            self.model.load_state_dict(state_dict)
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.image_processor = DetrImageProcessor.from_pretrained(pretrained)
        if train:
            self.dataloaders = create_dataloaders(
                config['training']['train_data_path'],
                config['training']['train_labels_path'],
                config['training']['val_data_path'],
                config['training']['val_labels_path'],
                batch_size=1,
                image_processor=self.image_processor
            )
        
        # Add SimplifiedTbNet
        self.d_model = self.model.config.hidden_size
        self.num_classes = 3
        self.num_nodes = 100
        self.simplified_tbnet = SimplifiedTbNet(self.d_model, self.num_classes, self.num_nodes)

    def forward(self, pixel_values, pixel_mask):
        detr_outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        probs, bbox_pairs = self.simplified_tbnet(detr_outputs.last_hidden_state, detr_outputs.pred_boxes)

        return probs, bbox_pairs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        detr_outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        
        probs, bbox_pairs = self.simplified_tbnet(detr_outputs.last_hidden_state, detr_outputs.pred_boxes)

        # here, we should calculate NLLLoss of probs against gt
        # parse through each bbox pair in bbox pairs and compare against the gt bboxes
        # and create a gt tensor of shape [batch size, num edges]

        

        loss = detr_outputs.loss
        loss_dict = detr_outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx) 
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
            
        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here: 
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return self.dataloaders[0]

    def val_dataloader(self):
        return self.dataloaders[1]