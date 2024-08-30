import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from data import create_dataloaders
from utils import state_dict_cleaner, depad, depad_2d, get_iou, convert_to_width_height

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
        # TODO where is the batch dimension of pred_bboxes??
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
            train_mode, with_gcn,
            pretrained, checkpoint,
            config,
            lr = 1e-4,
            lr_backbone = 1e-5,
            weight_decay = 1e-4):
        
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
        self.with_gcn = with_gcn
        
        # add SimplifiedTbNet
        if with_gcn:
            self.d_model, self.num_classes, self.num_nodes = self.model.config.hidden_size, 4, 100
            self.simplified_tbnet = SimplifiedTbNet(self.d_model, self.num_classes, self.num_nodes)
            self.gcn_criterion = nn.NLLLoss()

            # define the class integers here
            self.CLASS_NO_RELATION, self.CLASS_HORIZONTAL, self.CLASS_VERTICAL, self.CLASS_SAME_CELL = 0, 1, 2, 3

        if train_mode:
            self.dataloaders = create_dataloaders(
                config['training']['train_data_path'],
                config['training']['train_labels_path'],
                config['training']['val_data_path'],
                config['training']['val_labels_path'],
                batch_size=config['training']['batch_size'],
                image_processor=self.image_processor,
                with_gcn=self.with_gcn
            )
            self.model.train()
        else:
            self.model.eval()

    # expecting a cv2.imread numpy array. image can be batched
    def forward(self, image):

        target_sizes = torch.tensor([image.shape[:2]]).to(self.device)

        inputs = self.image_processor(images=image, return_tensors='pt').to(self.device)

        detr_outputs = self.model(**inputs)

        if self.with_gcn:

            results = self.image_processor.post_process_object_detection(
                outputs=detr_outputs,
                threshold=0,
                target_sizes=target_sizes
            )

            # to convert corner to coco bboxes. full explanation in utils.py
            pred_bboxes = torch.stack([convert_to_width_height(result['boxes']) for result in results])

            probs, bbox_pairs = self.simplified_tbnet(detr_outputs.last_hidden_state, pred_bboxes)
            # reminder: probs shape(batch_size, num_edges, num_classes),
            # bbox_pairs shape(batch_size, num_edges, 8) where bbox format is COCO
            return probs, bbox_pairs
        else:
            return detr_outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        detr_labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["detr_labels"]]

        detr_outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=detr_labels)
        
        detr_loss = detr_outputs.loss
        loss_dict = detr_outputs.loss_dict

        if self.with_gcn:

            gcn_labels = batch['gcn_labels']
            
            # detr_outputs.pred_boxes is normalized coords of format (centerx, centery, width, height)
            # we will use image processor to convert it to unnormalized corner (x0, y0, x1, y1)

            # prompt: detr_labels comes as a list of batch_size number of dicts. each dict has a key "orig_size", whose value is a tensor of shape (a, b). i need to collect all the orig_size values in detr_labels into a tensor of size (batch_size, a, b)
            batched_orig_sizes = torch.stack([label['orig_size'] for label in detr_labels])
            
            results = self.image_processor.post_process_object_detection(
                outputs=detr_outputs,
                threshold=0,
                target_sizes=batched_orig_sizes
            )

            # to convert corner to coco bboxes. full explanation in utils.py
            pred_bboxes = torch.stack([convert_to_width_height(result['boxes']) for result in results])

            probs, bbox_pairs = self.simplified_tbnet(detr_outputs.last_hidden_state, pred_bboxes)

            # convert gcn_labels from list(dict(shape(a,b))) to dict(shape(batch_size, a, b))
            batched_gcn_labels = {key: torch.stack([d[key] for d in gcn_labels]).to('cpu') for key in gcn_labels[0].keys()}
            gt = self.process_target(bbox_pairs, batched_gcn_labels['table_grid'], batched_gcn_labels['gt_bboxes'])
            gt = gt.to(self.device)

            # expect (batch_size, num_classes, *, *, ...)
            probs = probs.permute(0, 2, 1)
            gcn_loss = self.gcn_criterion(probs, gt)

            loss_dict['gcn_loss'] = gcn_loss
            loss = detr_loss + gcn_loss

            return loss, loss_dict
        
        else: return detr_loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)

        # Debugging print statement
        print(f"Batch {batch_idx} - Training Loss: {loss.item()}")
        for k, v in loss_dict.items():
            print(f"Batch {batch_idx} - {k}: {v.item()}")

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
    
    def process_target(self, bbox_pairs, table_grid, gt_bboxes):
        gt_bboxes, table_grid = depad(gt_bboxes), depad_2d(table_grid)

        # also, we should create the final gt object for assignment, a list of batch_size empty lists
        gt = [[] for _ in range(bbox_pairs.shape[0])]

        # now we can compare bbox pairs against gt bboxes based on some criterion
        # TODO this is super naive, consider using batched tensors with torchvision's box_iou tool
        # for now, since we're doing batches of 4 (max for 16gb card), this cpu method is cheap
        # compared to the amt of time the batch stays in the gpu
        bbox_pairs = bbox_pairs.tolist()
        for batch_index, batch in enumerate(bbox_pairs):
            for edge in batch:
                # let's split the edge data into bbox 1 and bbox 2
                bbox1, bbox2 = edge[:4], edge[4:]
                # let's try to get the corresponding gt bboxes for each bbox in bbox pairs
                max1, max2, index1, index2 = 0, 0, None, None
                for gt_bbox_index, gt_bbox in enumerate(gt_bboxes[batch_index]):
                    iou1, iou2 = get_iou(bbox1, gt_bbox), get_iou(bbox2, gt_bbox)
                    if iou1 > max1:
                        max1 = iou1
                        index1 = gt_bbox_index
                    if iou2 > max2:
                        max2 = iou2
                        index2 = gt_bbox_index

                # in this final stage, we will get all of the gt classes based on our table grid rules
                gt_class = None

                # here we must assign the groundtruth to NO RELATION class if either bbox is undefined
                if max1 < 0.5 or max2 < 0.5:
                    gt_class = self.CLASS_NO_RELATION
                    gt[batch_index].append(gt_class)
                    continue

                if index1 == index2:
                    gt_class = self.CLASS_SAME_CELL
                    gt[batch_index].append(gt_class)
                    continue

                # parse through the table grid for VERTICAL or HORIZONTAL relations
                table_grid_sample = table_grid[batch_index]
                for row_index, row in enumerate(table_grid_sample):
                    for col_index, item in enumerate(row):
                        if item == index1:
                            # first bbox locked in, let's do a column sweep for VERTICAL relation
                            for sweep_index in range(len(table_grid_sample)):
                                if table_grid_sample[sweep_index][col_index] == index2:
                                    gt_class = self.CLASS_VERTICAL
                                    break
                            # if the above doesn't work, let's do a rowsweep for HORIZONTAL relation
                            for sweep_index in range(len(row)):
                                if table_grid_sample[row_index][sweep_index] == index2:
                                    gt_class = self.CLASS_HORIZONTAL
                                    break
                        if gt_class: break
                    if gt_class: break
                
                # the 2 bboxes are gt-matched, of different gt index and no vert or horiz relation.
                # thus we conclude there must be no relation
                if not gt_class: gt_class = self.CLASS_NO_RELATION

                gt[batch_index].append(gt_class)

        return torch.tensor(gt, dtype=torch.long)