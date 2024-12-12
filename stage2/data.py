import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from utils import get_table_grid, process_target, get_psuedo_knn, get_feature_vec
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

class TrainDataset(Dataset):

    # now implemented with jsonl lazy loading! great success!
    def __init__(self, label_path):
        self.label_path = label_path
        self.line_offsets = self._get_line_offsets()

    def _get_line_offsets(self):
        offsets = []
        with open(self.label_path, 'r', encoding="utf8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                offsets.append(offset)
        return offsets

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        offset = self.line_offsets[idx]
        with open(self.label_path, 'r', encoding="utf8") as f:
            f.seek(offset)
            line = f.readline()
            label = json.loads(line)

        # first, we get the table grid of the html
        thead_grid, tbody_grid = get_table_grid(''.join(label['html']))
        table_grid = thead_grid + tbody_grid

        bboxes = torch.tensor(label['bboxes'], dtype=torch.int)
        bbox_indices = torch.tensor(label['bbox_indices'], dtype=torch.int)

        edge_index = get_psuedo_knn(bboxes)

        bbox_index_pairs = torch.stack((
            bbox_indices[edge_index[0]],  # Start bounding boxes
            bbox_indices[edge_index[1]]   # End bounding boxes
        ), dim=1)

        gt_classes = process_target(bbox_index_pairs, table_grid) # dtype long

        feature_vec = get_feature_vec(bboxes, torch.tensor([label['orig_size']['width'],
                                       label['orig_size']['height']], dtype=torch.float32))
        
        data = Data(x=feature_vec, edge_index=edge_index)

        max_length = 10000

        # pyg dataloader overrides collate fn, so cannot use pad_sequence there
        # foresee excceeding 10000 is very very rare
        if gt_classes.size(0) > max_length:
            gt_classes = gt_classes[:max_length]

        gt_classes = F.pad(gt_classes, (0, max_length - gt_classes.size(0)), value=-1)
        
        return data, gt_classes

def create_dataloaders(train_label_path, val_label_path, batch_size=8, num_workers=8):
    train_dataset = TrainDataset(train_label_path)
    val_dataset = TrainDataset(val_label_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader