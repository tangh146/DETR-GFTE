import json
import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_table_grid

class TrainDataset(Dataset):

    # now implemented with jsonl lazy loading! great success!
    def __init__(self, label_path):
        """
        Args:
            img_dir (string): Directory with all the images.
            label_file (string): Path to the jsonl file with annotations.
        """
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
        offset = self.line_offsets[0]
        with open(self.label_path, 'r', encoding="utf8") as f:
            f.seek(offset)
            line = f.readline()
            label = json.loads(line)

        filename = label['filename']

        # first, we get the table grid of the html
        thead_grid, tbody_grid = get_table_grid(''.join(label['html']))
        table_grid = thead_grid + tbody_grid

        gcn_label = {
            'table_grid': torch.tensor(table_grid, dtype=torch.int).to('cuda'),
            'gt_bboxes': torch.tensor(label['bboxes'], dtype=torch.float32).to('cuda'),
            'gt_bbox_indices': torch.tensor(label['bbox_indices'], dtype=torch.int).to('cuda'),
            'orig_size': torch.tensor([label['orig_size']['width'],
                                       label['orig_size']['height']], dtype=torch.float32).to('cuda')
        }
        
        return gcn_label

def create_dataloaders(train_label_path, val_label_path, batch_size=None):
    train_dataset = TrainDataset(train_label_path)
    val_dataset = TrainDataset(val_label_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader