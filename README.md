# DETR-GFTE Graph-based Table Recognition

Experimental graph-based table recognition models based on [DETR](https://arxiv.org/abs/2005.12872) and [GFTE](https://arxiv.org/abs/2003.07560).

![Example input](https://github.com/tangh146/DETR-GFTE/blob/main/misc/PMC4682394_003_00.png?raw=true)
![Example output](https://github.com/tangh146/DETR-GFTE/blob/main/misc/demo_output.png?raw=true)

Do read our [report](https://github.com/tangh146/DETR-GFTE/blob/main/model_report.pdf) for more information.

## Usage

### 1. Set up environment

Set up your environment and install `requirements.txt`.

### 2. Download checkpoints folder

Download the checkpoints folder [here](https://drive.google.com/drive/folders/1luUrVkRi4txh5Pt_TSnyh3Jg3x26oAXh?usp=sharing) and place the folder in DETR-GFTE directory.

### 3. Run the `inference.ipynb` notebooks

You may now run the `inference.ipynb` notebooks.

### 4. Run `evaluation.ipynb` and `trainer.ipynb` notebooks

To run the `evaluation.ipynb` and `trainer.ipynb` notebooks, download the PubTabNet dataset [here](https://developer.ibm.com/exchanges/data/all/pubtabnet/) and follow the instructions in the notebooks to prepare the `.jsonl` annotation files.
