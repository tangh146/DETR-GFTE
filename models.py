import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from data import create_dataloaders
from utils import state_dict_cleaner, depad, depad_2d, get_iou, decoco, get_psuedo_knn
import supervision as sv

class SimplifiedTbNet(nn.Module):
    def __init__(self, d_model, num_classes):
        super(SimplifiedTbNet, self).__init__()
        self.conv1 = GCNConv(d_model, d_model)
        self.conv2 = GCNConv(d_model, d_model)
        self.lin1 = torch.nn.Linear(d_model * 2, d_model)
        self.lin_final = torch.nn.Linear(d_model, num_classes)

    def forward(self, hidden_state, pred_bboxes):
        # hidden_state: shape(num nodes, d model)
        # pred bboxes: shape(num_nodes, 4)

        edge_index = get_psuedo_knn(pred_bboxes).to(hidden_state.device)

        # GCN Layers
        x = self.conv1(hidden_state, edge_index)
        # print("After conv1:", x)
        x = F.relu(x)
        # print("After ReLU1:", x)

        x = self.conv2(x, edge_index)
        # print("After conv2:", x)
        x = F.relu(x)
        # print("After ReLU2:", x)

        # Pairing Node Features
        x1 = x[edge_index[0]]
        x2 = x[edge_index[1]]
        xpair = torch.cat((x1, x2), dim=-1)  # Shape: [num_edges, d_model * 2]
        xpair = F.relu(self.lin1(xpair))

        # Final Classification
        xfin = self.lin_final(xpair)
        probs = F.log_softmax(xfin, dim=-1)  # Shape: [num_edges, num_classes]

        # Process Bounding Boxes
        bbox_pairs = torch.cat((
            pred_bboxes[edge_index[0]],  # Start bounding boxes
            pred_bboxes[edge_index[1]]   # End bounding boxes
        ), dim=-1)

        # bbox pairs shape = [num edges, bbox dim * 2]
        # probs shape = [num edges, num classes]

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
            num_labels=config['training']['num_detr_classes'],
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

        if train_mode:
            self.dataloaders = create_dataloaders(
                config['training']['train_data_path'],
                config['training']['train_labels_path'],
                config['training']['val_data_path'],
                config['training']['val_labels_path'],
                batch_size=config['training']['batch_size'],
                image_processor=self.image_processor,
                with_gcn=self.with_gcn)
        
        if with_gcn:
            # FREEZE ALL WEIGHTS OF ENCODER
            for param in self.model.parameters():
                param.requires_grad = False

            # TODO move all this crap out as well
            self.d_model, self.num_classes = self.model.config.hidden_size, 3
            self.simplified_tbnet = SimplifiedTbNet(self.d_model, self.num_classes)
            self.gcn_criterion = nn.NLLLoss()

            # define the class integers here
            self.CLASS_NO_RELATION, self.CLASS_HORIZONTAL, self.CLASS_VERTICAL = 0, 1, 2
        
        self.bbox_encoder = BBoxEncoder()

        if self.train and not self.with_gcn:
            self.model.train()
        else:
            self.model.eval()
        
        self.simplified_tbnet.train()
        self.bbox_encoder.train()

    # expecting a cv2.imread numpy array
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

            for result in results:
                detections = sv.Detections.from_transformers(transformers_results=result).with_nms(threshold=0)
                pred_bboxes = torch.tensor(decoco(detections.xyxy), dtype=torch.float32).to(self.device)
                print(f'pred_bboxes: {pred_bboxes.tolist()}')
                hidden_states = self.bbox_encoder(pred_bboxes)

                probs, bbox_pairs = self.simplified_tbnet(hidden_states, pred_bboxes)

                break

            # reminder: probs shape(batch size, num_edges, num_classes),
            # bbox_pairs shape(batchsize, num_edges, 8) where bbox format is COCO
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
            
            # results = self.image_processor.post_process_object_detection(
            #     outputs=detr_outputs,
            #     threshold=0,
            #     target_sizes=batched_orig_sizes
            # )

            singular_losses = []
            for index in range(len(gcn_labels)):
                gt_bboxes = torch.tensor(depad(gcn_labels[index]['gt_bboxes'].to('cpu')),
                                         dtype=torch.float32).to(self.device)
                # Define the value to check
                target_values = torch.tensor([0.0, 0.0, 0.01, 0.01]).to(self.device)

                # Find rows that do NOT match the target values
                mask = ~(gt_bboxes == target_values).all(dim=1)

                # Filter out rows that match the target values
                cheat_bboxes = gt_bboxes[mask]

                hidden_states = self.bbox_encoder(cheat_bboxes)

                probs, bbox_pairs = self.simplified_tbnet(hidden_states, cheat_bboxes)

                gt = self.process_target(bbox_pairs,
                                            gcn_labels[index]['table_grid'].to('cpu'), 
                                            gcn_labels[index]['gt_bboxes'].to('cpu'))
                # expect (batch_size, num_classes, *, *, ...)
                probs = probs.unsqueeze(0)
                gt = gt.unsqueeze(0)
                probs = probs.permute(0, 2, 1)
                singular_losses.append(self.gcn_criterion(probs, gt))

            # singular_losses = []
            # for index, result in enumerate(results):
            #     detections = sv.Detections.from_transformers(transformers_results=result).with_nms(threshold=0)
            #     pred_bboxes = torch.tensor(decoco(detections.xyxy), dtype=torch.float32).to(self.device)

            #     hidden_states = self.bbox_encoder(pred_bboxes)

            #     probs, bbox_pairs = self.simplified_tbnet(hidden_states, pred_bboxes)

            #     gt = self.process_target(bbox_pairs,
            #                                 gcn_labels[index]['table_grid'].to('cpu'), 
            #                                 gcn_labels[index]['gt_bboxes'].to('cpu'))
            #     # expect (batch_size, num_classes, *, *, ...)
            #     probs = probs.unsqueeze(0)
            #     gt = gt.unsqueeze(0)
            #     probs = probs.permute(0, 2, 1)
            #     singular_losses.append(self.gcn_criterion(probs, gt))

            gcn_loss = torch.mean(torch.stack(singular_losses))

            loss_dict = {'gcn_loss': gcn_loss}

            return gcn_loss, loss_dict
        
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

        gt = []

        # now we can compare bbox pairs against gt bboxes based on some criterion
        # TODO this is super naive, consider using batched tensors with torchvision's box_iou tool
        # for now, since we're doing batches of 4 (max for 16gb card), this cpu method is cheap
        # compared to the amt of time the batch stays in the gpu
        bbox_pairs = bbox_pairs.tolist()
        for edge in bbox_pairs:
            # let's split the edge data into bbox 1 and bbox 2
            bbox1, bbox2 = edge[:4], edge[4:]
            # let's try to get the corresponding gt bboxes for each bbox in bbox pairs
            max1, max2, index1, index2 = 0, 0, None, None
            for gt_bbox_index, gt_bbox in enumerate(gt_bboxes):
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
            if max1 < 0.1 or max2 < 0.1:
                gt_class = self.CLASS_NO_RELATION
                gt.append(gt_class)
                continue

            # parse through the table grid for VERTICAL or HORIZONTAL relations
            # TODO figure out if multi spanning cells are accounted for
            for row_index, row in enumerate(table_grid):
                for col_index, item in enumerate(row):
                    if item == index1:
                        # first bbox locked in, let's do a column sweep for VERTICAL relation
                        for sweep_index in range(len(table_grid)):
                            if table_grid[sweep_index][col_index] == index2:
                                gt_class = self.CLASS_VERTICAL
                                break
                        # if the above doesn't work, let's do a rowsweep for HORIZONTAL relation
                        for sweep_index in range(len(row)):
                            if table_grid[row_index][sweep_index] == index2:
                                gt_class = self.CLASS_HORIZONTAL
                                break
                    if gt_class: break
                if gt_class: break
            
            # the 2 bboxes are gt-matched, of different gt index and no vert or horiz relation.
            # thus we conclude there must be no relation
            if not gt_class: gt_class = self.CLASS_NO_RELATION

            gt.append(gt_class)

        return torch.tensor(gt, dtype=torch.long).to(self.device)

# Define the encoding network
class BBoxEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=256):
        super(BBoxEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x