import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class TrainDataset(Dataset):

    # now implemented with jsonl lazy loading! great success!
    def __init__(self, data_path, label_path, image_processor):
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
        
        encoding = self.image_processor(images = image, annotations=label, return_tensors='pt')
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

def collate_fn(batch, image_processor):
# DETR authors employ various image sizes during training, making it not possible 
# to directly batch together images. Hence they pad the images to the biggest 
# resolution in a given batch, and create a corresponding binary pixel_mask 
# which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

def create_dataloaders(train_data_path, train_label_path, val_data_path, val_label_path, batch_size, image_processor):
    train_dataset = TrainDataset(train_data_path, train_label_path, image_processor)
    val_dataset = TrainDataset(val_data_path, val_label_path, image_processor)

    train_loader = DataLoader(train_dataset, collate_fn = lambda b: collate_fn(b, image_processor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, collate_fn = lambda b: collate_fn(b, image_processor), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader