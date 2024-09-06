import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import get_table_grid, pad, pad_2d

class TrainDataset(Dataset):

    # now implemented with jsonl lazy loading! great success!
    def __init__(self, data_path, label_path, image_processor, with_gcn):
        """
        Args:
            img_dir (string): Directory with all the images.
            label_file (string): Path to the jsonl file with annotations.
        """
        self.data_path = data_path
        self.label_path = label_path
        # TODO checkpoint? what if i am finetuning
        # self.image_processor = DetrImageProcessor.from_pretrained(checkpoint)
        self.image_processor = image_processor
        self.with_gcn = with_gcn
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

        filename = label['filename']
        img_path = os.path.join(self.data_path, filename)
        image = Image.open(img_path).convert("RGB")

        encoder_annotations = []
        for annotation in label['annotations']:
            if annotation['bbox'] != [0,0,0.01,0.01]:
                encoder_annotations.append(annotation)

        # get the dict of tensorized GCN labels
        if self.with_gcn:
            # first, we get the table grid of the html
            thead_grid, tbody_grid = get_table_grid(''.join(label['html']))
            table_grid = thead_grid + tbody_grid

            # pad the table grid to a side len of 40 and padding token -1
            table_grid = pad_2d(table_grid, pad_to=89, padding_token=-1)

            # a very janky way to separate encoder from gcn annotations.
            #  gcn requires empty bbox indicators while encoder doesnt
            gt_bboxes= []
            for annotation in label['annotations']:
                gt_bboxes.append(annotation['bbox'])

            gt_bboxes = pad(gt_bboxes, pad_to=2061, padding_token=[-1,-1,-1,-1])

            gcn_label = {
                'table_grid': torch.tensor(table_grid, dtype=torch.float32),
                'gt_bboxes': torch.tensor(gt_bboxes, dtype=torch.float32)
            }
        
        label['annotations'] = encoder_annotations
        
        encoding = self.image_processor(images = image, annotations=label, return_tensors='pt')

        pixel_values = encoding["pixel_values"].squeeze()
        # get the dict of tensorized DETR labels
        detr_label = encoding["labels"][0]

        if self.with_gcn:
            return pixel_values, detr_label, gcn_label
        else: return pixel_values, detr_label

def collate_fn(batch, image_processor, with_gcn):
# DETR authors employ various image sizes during training, making it not possible 
# to directly batch together images. Hence they pad the images to the biggest 
# resolution in a given batch, and create a corresponding binary pixel_mask 
# which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    detr_labels = [item[1] for item in batch]
    if with_gcn:
        gcn_labels = [item[2] for item in batch]
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'detr_labels': detr_labels,
            'gcn_labels': gcn_labels
        }
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'detr_labels': detr_labels,
    }

def create_dataloaders(train_data_path, train_label_path, val_data_path, val_label_path, batch_size, image_processor, with_gcn):
    train_dataset = TrainDataset(train_data_path, train_label_path, image_processor, with_gcn=with_gcn)
    val_dataset = TrainDataset(val_data_path, val_label_path, image_processor, with_gcn=with_gcn)

    train_loader = DataLoader(train_dataset, collate_fn = lambda b: collate_fn(b, image_processor, with_gcn=with_gcn), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, collate_fn = lambda b: collate_fn(b, image_processor, with_gcn=with_gcn), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader