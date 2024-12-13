# DETR-GFTE Graph-based Table Recognition

Experimental graph-based table recognition models based on [DETR](https://arxiv.org/abs/2005.12872) and [GFTE](https://arxiv.org/abs/2003.07560).

## Usage

1.Set up environment 
git clone  https://github.com/tangh146/DETR-GFTE.git
cd DETR-GFTE
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

2.Download checkpoints folder
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1luUrVkRi4txh5Pt_TSnyh3Jg3x26oAXh
Alternatively:
[Download the checkpoints folder](https://drive.google.com/drive/folders/1luUrVkRi4txh5Pt_TSnyh3Jg3x26oAXh?usp=sharing) and place the folder in DETR-GFTE directory.

3.You may now run the `inference.ipynb` notebooks. 


4.To run the `evaluation.ipynb` and `training.ipynb` notebooks, [download the PubTabNet dataset](https://developer.ibm.com/exchanges/data/all/pubtabnet/) and follow the instructions in the notebooks to prepare the `.jsonl` annotation files.
