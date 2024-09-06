import yaml
import torch
from bs4 import BeautifulSoup as bs
from typing import List
import cv2
import math

def load_config(config_file):
    """
    Load the configuration from a YAML file.
    
    Args:
    config_file (str): Path to the YAML configuration file.
    
    Returns:
    dict: Configuration dictionary.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def euclidean_distance(p1, p2):
    """
    Compute the Euclidean distance between two points.
    
    Args:
    p1 (tuple): First point.
    p2 (tuple): Second point.
    
    Returns:
    float: Euclidean distance between p1 and p2.
    """
    return torch.sqrt(torch.sum((torch.tensor(p1) - torch.tensor(p2)) ** 2))

# gets the cell indices (wrt to all cells) of the header cells that touch the table body, for the purpose of extracting their embeddings
def get_table_grid(html):
    soup = bs(html, 'html.parser')
    thead = soup.find('thead')
    tbody = soup.find('tbody')

    # Extract all rows in the thead
    thead_rows = thead.find_all('tr')
    tbody_rows = tbody.find_all('tr')

    def get_segment_grid(segm_rows, current_cell = 0):
        # contains tuples (cell number, remaining rowspan) of the current row in the iteration
        row_cursor = []
        # memory grid
        grid = []

        # iterate through the rows
        for row_idx in range(len(segm_rows)):
            current_row = segm_rows[row_idx].find_all(['td'])
            for cell_idx in range(len(current_row)):
                rowspan = int(current_row[cell_idx].get('rowspan', 1))
                colspan = int(current_row[cell_idx].get('colspan', 1))

                if row_idx == 0:
                    # for first row, populate row_cursor
                    for i in range(colspan):
                        row_cursor.append((current_cell, rowspan))
                    current_cell += 1
                else:
                    # for all subsequent rows, find an empty block in row_cursor that accommodates the current cell's colspan
                    count = 0
                    start_index = 0
                    for i in range(len(row_cursor)):
                        if row_cursor[i][1] == 0:
                            if count == 0:
                                start_index = i  # Set the start index of the sequence
                            count += 1
                            if count == colspan:
                                # deal with any valid block successfully found
                                for j in range(start_index, start_index + colspan):
                                    row_cursor[j] = (current_cell, rowspan)
                                current_cell += 1
                                break
                        else:
                            count = 0  # Reset count if non-zero is encountered
            # print(current_cell)
            # print(row_cursor)
            # save the current row cursor indices to the memory grid
            grid.append([tup[0] for tup in row_cursor])
            # decrement the previous row rowspans when advancing to the bext row
            for tup_idx in range(len(row_cursor)):
                cell, rowspan = row_cursor[tup_idx]
                row_cursor[tup_idx] = (cell, rowspan - 1)

        return grid, current_cell
            
    thead_grid, current_cell = get_segment_grid(thead_rows)
    tbody_grid, _ = get_segment_grid(tbody_rows, current_cell)

    # extract the indices of the row cursor in its final state
    # bottommost_indices = [tup[0] for tup in row_cursor]

    return thead_grid, tbody_grid

# collates individual bbox coord embeddings and averages them
def average_consecutive_sets_of_four(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    averaged_tensors = []
    for i in range(0, len(tensors) - len(tensors) % 4, 4):
        set_of_four = tensors[i:i + 4]
        average_tensor = torch.mean(torch.stack(set_of_four), dim=0)
        averaged_tensors.append(average_tensor)
    return averaged_tensors

# remove extra "model" prefixes from huggingface detr state dict keys
def state_dict_cleaner(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict

def pad_2d(original_list, pad_to=50, padding_token=-1):
    # Get the dimensions of the original list
    rows = len(original_list)
    cols = len(original_list[0]) if rows > 0 else 0

    # Create a new 50x50 list filled with -1
    padded_list = [[padding_token for _ in range(pad_to)] for _ in range(pad_to)]

    # Copy the original list into the padded list
    for i in range(min(rows, pad_to)):
        for j in range(min(cols, pad_to)):
            padded_list[i][j] = original_list[i][j]

    return padded_list

def depad_2d(padded_tensor, padding_token=-1):
    sample_list = []
    for row in padded_tensor:
        # Convert to list and remove padding
        row_list = [val.item() for val in row if val != padding_token]
        if row_list:  # Only add non-empty rows
            sample_list.append(row_list)
    return sample_list

def pad(original_list, pad_to=100, padding_token=[-1, -1, -1, -1]):
    # Create a new list with the target length
    padded_list = original_list.copy()
    
    # Pad the list to the target length
    while len(padded_list) < pad_to:
        padded_list.append(padding_token)
    
    return padded_list

def depad(padded_tensor, padding_token=[-1, -1, -1, -1]):
    padding_token = torch.tensor(padding_token, dtype=padded_tensor.dtype)
    
    sample_list = []
    for row in padded_tensor:
        if not torch.equal(row, padding_token):
            sample_list.append(row.tolist())
    
    return sample_list

# coco format to normal format
def convert_bbox_format(bbox):
    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height
    return [xmin, ymin, xmax, ymax]

# source: http://ronny.rest/tutorials/module/localization_001/iou/
# in the ideal scenario, you should calculate iou scores as batched tensors
# using the torchvision box_iou method to relieve cpu 
def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    a = convert_bbox_format(a)
    b = convert_bbox_format(b)

    # Check if bbox `a` is completely within bbox `b`
    if (a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]):
        return 1.0
    
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

# convert xyxy to coco bbox format
def decoco(bboxes):
    converted_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        converted_bboxes.append([xmin, ymin, width, height])
    return converted_bboxes

# prompt and unit test in gencode_testbed
def draw_bboxes_and_edges(image, prob_tensor, edge_tensor, bbox_thickness=2, line_thickness=2):
    # Define the color mapping for the classes
    colors = {
        1: (0, 0, 255),   # Red
        2: (255, 0, 0)   # Blue
    }

    batch_size, num_edges, num_classes = prob_tensor.shape

    # Ensure prob_tensor is in log-softmax format and get the predicted class per edge
    predicted_classes = torch.argmax(prob_tensor, dim=-1)

    for i in range(batch_size):
        for j in range(num_edges):
            # Extract the bounding boxes for the current edge
            bbox1 = edge_tensor[i, j, :4].cpu().numpy().astype(int)
            bbox2 = edge_tensor[i, j, 4:].cpu().numpy().astype(int)

            # Draw the bounding boxes on the image
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2

            cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 255), bbox_thickness)
            cv2.rectangle(image, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 255), bbox_thickness)

            # Get the predicted class and corresponding color
            predicted_class = predicted_classes[i, j].item()
            if predicted_class == 0:
                continue  # Skip drawing for class 0

            color = colors[predicted_class]

            # Draw the edge as a line connecting the centers of the two bounding boxes
            center1 = (x1 + w1 // 2, y1 + h1 // 2)
            center2 = (x2 + w2 // 2, y2 + h2 // 2)
            cv2.line(image, center1, center2, color, line_thickness)

    return image

def intersect_1d(start1, end1, start2, end2):
    return max(start1, start2) <= min(end1, end2)

def get_psuedo_knn(bboxes, default_radius=30, candidate_radius = 50):
    """
    Construct a pseudo-KNN graph based on the bounding box overlap and centroids.
    
    Args:
        bboxes (torch.Tensor): Tensor of shape (num_bboxes, 4) where each row represents
                               [x_min, y_min, width, height] for a bounding box.
        default_radius (int): Radius used to create pseudo connections when no neighbors are found.

    Returns:
        edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges) representing the connections.
    """
    # Initialize the edge index list to store connections
    edge_index = torch.empty((2, 0), dtype=torch.long)

    # Compute the centroids of all bounding boxes
    centers = bboxes[:, :2] + (bboxes[:, 2:] / 2)  # Shape: (num_bboxes, 2)
    num_bboxes = bboxes.size(0)

    # Iterate over each target bbox to construct its neighborhood
    for i, target_bbox in enumerate(bboxes):
        horz_candidates, vert_candidates = [], []

        for j, knn_bbox in enumerate(bboxes):
            if intersect_1d(target_bbox[0] - candidate_radius, target_bbox[0] + target_bbox[2] + candidate_radius, 
                            knn_bbox[0], knn_bbox[0] + knn_bbox[2]):
                vert_candidates.append(knn_bbox)
            elif intersect_1d(target_bbox[1] - candidate_radius, target_bbox[1] + target_bbox[3] + candidate_radius, 
                              knn_bbox[1], knn_bbox[1] + knn_bbox[3]):
                horz_candidates.append(knn_bbox)
        
        # Initialize nearest neighbors
        up, down, left, right = None, None, None, None
        
        # Horizontal candidates (left and right)
        for horz_candidate in horz_candidates:
            # left candidate
            if horz_candidate[0] + horz_candidate[2] < target_bbox[0]:
                if left is None or horz_candidate[0] + horz_candidate[2] > left[0] + left[2]:
                    left = horz_candidate
            # right candidate
            if horz_candidate[0] > target_bbox[0] + target_bbox[2]:
                if right is None or horz_candidate[0] < right[0]:
                    right = horz_candidate
        
        # Vertical candidates (up and down)
        for vert_candidate in vert_candidates:
            # up candidate
            if vert_candidate[1] + vert_candidate[3] < target_bbox[1]:
                if up is None or vert_candidate[1] + vert_candidate[3] > up[1] + up[3]:
                    up = vert_candidate
            # down candidate
            if vert_candidate[1] > target_bbox[1] + target_bbox[3]:
                if down is None or vert_candidate[1] < down[1]:
                    down = vert_candidate
        
        # Fallback to default radius if no neighbors found
        if left is None: left = [target_bbox[0] - default_radius, 0, 0, 0]
        if right is None: right = [target_bbox[0] + default_radius, 0, target_bbox[2], 0]
        if up is None: up = [0, target_bbox[1] - default_radius, 0, 0]
        if down is None: down = [0, target_bbox[1] + default_radius, 0, target_bbox[3]]

        # Construct the local area of interest (bounding box)
        xmin, xmax = left[0], right[0] + right[2]
        ymin, ymax = up[1], down[1] + down[3]

        # Now, find all bboxes whose centroids lie within the area of interest
        for j, centroid in enumerate(centers):
            if xmin <= centroid[0] <= xmax and ymin <= centroid[1] <= ymax:
                # Create an edge between the target bbox and the neighboring bbox
                edge_index = torch.cat([edge_index, torch.tensor([[i], [j]])], dim=1)

    return prune_and_bidirectional(edge_index)

# normalizes coco bboxes and returns feature vector
def get_feature_vec(gt_bboxes, orig_size):
    # Original image width and height
    orig_w, orig_h = orig_size[0], orig_size[1]
    
    # Extract bbox information
    x1 = gt_bboxes[:, 0] / orig_w  # Top-left x (normalized)
    y1 = gt_bboxes[:, 1] / orig_h  # Top-left y (normalized)
    x2 = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / orig_w  # Bottom-right x (normalized)
    y2 = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / orig_h  # Bottom-right y (normalized)
    
    # Compute center points
    cx = (x1 + x2) / 2  # Center x
    cy = (y1 + y2) / 2  # Center y
    
    # Compute width and height (normalized)
    w = (x2 - x1)  # Normalized width
    h = (y2 - y1)  # Normalized height
    
    # Stack all features into a feature vector of size 8
    feature_vector = torch.stack([x1, x2, y1, y2, cx, cy, w, h], dim=1)
    
    return feature_vector

def prune_and_bidirectional(edge_index):
    # Step 1: Prune self-connections
    mask = edge_index[0] != edge_index[1]  # Remove self-connections
    edge_index = edge_index[:, mask]        # Apply mask to prune self-loops

    # Step 2: Convert unidirectional edges to bidirectional
    # Convert to a set of tuples for fast lookup
    edge_set = set(map(tuple, edge_index.t().tolist()))

    # List to hold new bidirectional edges
    new_edges = []

    # Check for unidirectional edges and add reverse if not present
    for i, j in edge_set:
        if (j, i) not in edge_set:  # If reverse edge does not exist
            new_edges.append([j, i])

    # Add the new bidirectional edges
    if new_edges:
        new_edges_tensor = torch.tensor(new_edges, device=edge_index.device).t()
        edge_index = torch.cat([edge_index, new_edges_tensor], dim=1)

    return edge_index

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe