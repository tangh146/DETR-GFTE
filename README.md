# DETR-GFTE Graph-based Table Recognition

Experimental graph-based table recognition models based on [DETR](https://arxiv.org/abs/2005.12872) and [GFTE](https://arxiv.org/abs/2003.07560).

## Usage

### 1. Set up environment 

```bash
git clone  https://github.com/tangh146/DETR-GFTE.git

cd DETR-GFTE

python3 -m venv myenv

# depending on whether you use ubuntu or windows
source myenv/bin/activate or myenv\Scripts\activate

# download and install requirements
pip install -r requirements.txt
```

### 2. Download checkpoints folder

```bash
pip install gdown

gdown --folder https://drive.google.com/drive/folders/1luUrVkRi4txh5Pt_TSnyh3Jg3x26oAXh
```

Or alternatively:

[Download the checkpoints folder](https://drive.google.com/drive/folders/1luUrVkRi4txh5Pt_TSnyh3Jg3x26oAXh?usp=sharing) and place the folder in DETR-GFTE directory.

### 3. Run the `inference.ipynb` notebooks
If you are using Ubuntu, in the code, replace any Windows-style file paths (e.g., \\) with Linux-style paths that use forward slashes (/).

### 4. Run `evaluation.ipynb` and `trainer.ipynb` notebooks
To run the `evaluation.ipynb` and `trainer.ipynb` notebooks, [download the PubTabNet dataset](https://developer.ibm.com/exchanges/data/all/pubtabnet/) and follow the instructions in the notebooks to prepare the `.jsonl` annotation files.