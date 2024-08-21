from utils import get_table_grid
import cv2
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms

def divide_image_into_patches(image, patch_size):
    pxpy = []
    patches = []
    h, w, _ = image.shape
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            pxpy.append((x, y))
            patches.append(patch)
    return pxpy, patches

def calculate_intersection_area(patch_coords, bbox):
    px, py, patch_size = patch_coords
    xmin, ymin, xmax, ymax = bbox

    # Calculate the coordinates of the intersection rectangle
    inter_xmin = max(px, xmin)
    inter_ymin = max(py, ymin)
    inter_xmax = min(px + patch_size, xmax)
    inter_ymax = min(py + patch_size, ymax)

    # Calculate the area of the intersection rectangle
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    return inter_area

def calculate_intersection_scores(image, bboxes, patch_size=16):
    # Convert PIL Image to NumPy array
    image = np.array(image)
    pxpy, _ = divide_image_into_patches(image, patch_size)
    patch_area = patch_size * patch_size
    patch_scores = []

    for px, py in pxpy:
        patch_dict = {}
        patch_coords = (px, py, patch_size)
        
        for bbox, header_index in bboxes.items():
            inter_area = calculate_intersection_area(patch_coords, bbox)
            if inter_area > 0:
                intersection_score = inter_area / patch_area
                patch_dict[header_index] = intersection_score
        
        patch_scores.append(patch_dict)
    
    return patch_scores

def flatten_patch_scores(patch_scores):
    flattened_scores = []
    for scores in patch_scores:
        if not scores:
            flattened_scores.append(0)
        else:
            # Find the key with the highest value
            best_header_index = max(scores, key=scores.get)
            flattened_scores.append(best_header_index)
    return flattened_scores

# takes the raw pubtabnet label, returns a dict containing the GT non-empty bbox centroids as keys and the assoc GT header indices as values
def preprocess_data(image, label, patch_size, target_size):
    html = label['html']['structure']['tokens'].copy()
    bboxes = []

    cell_idx = 0
    non_empty_indices = []

    # Traverse the html structure tokens to find <td> tags
    for i in range(len(html)):
        if html[i] in ['<td>', '<td']:
            # Check if there are more cells to process
            if cell_idx < len(label['html']['cells']):
                cell = label['html']['cells'][cell_idx]
                if 'bbox' in cell:
                    # Calculate bbox centroid
                    bboxes.append(cell['bbox'])
                    # insert []
                    # html.insert(i + 1, "[]")
                    non_empty_indices.append(cell_idx)
                cell_idx += 1
    
    resized_image = image.resize((target_size, target_size), Image.LANCZOS)
    
    # Calculate the ratios
    original_width, original_height = image.size
    ratio_w = target_size / original_width
    ratio_h = target_size / original_height

    # Adjust the bounding boxes
    resized_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        xmin = int(xmin * ratio_w)
        ymin = int(ymin * ratio_h)
        xmax = int(xmax * ratio_w)
        ymax = int(ymax * ratio_h)
        resized_bboxes.append((xmin, ymin, xmax, ymax))

    html = ''.join(html)

    gt_thead_grid, gt_tbody_grid = get_table_grid(html)
    gt_table_grid = gt_thead_grid + gt_tbody_grid
    
    # extract the groundtruth table header indices where there are non empty cells, so as to pair them with the gt bbox centroids in a dict
    gt_thead_indices = [0] * len(non_empty_indices)
    for row in gt_table_grid:
        for i in range(len(row) -1,-1,-1):
            for j in range(len(non_empty_indices)):
                if row[i] == non_empty_indices[j]:
                    # rare errors from specific tables. a pubtabnet issue, not mine
                    try:
                        gt_thead_indices[j] = i+1
                    except:
                        pass

    gt_bbox_dict = {tuple(key): value for key, value in zip(resized_bboxes, gt_thead_indices)}

    gt_patch_scores = flatten_patch_scores(calculate_intersection_scores(image=resized_image, bboxes=gt_bbox_dict, patch_size=patch_size))
    
    preprocess = transforms.ToTensor()
    resized_image = preprocess(resized_image)

    
    return resized_image, gt_patch_scores