import yaml
import torch
from bs4 import BeautifulSoup as bs
from typing import List
import cv2

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
    depadded_list = []
    for sample in padded_tensor:
        sample_list = []
        for row in sample:
            # Convert to list and remove padding
            row_list = [val.item() for val in row if val != padding_token]
            if row_list:  # Only add non-empty rows
                sample_list.append(row_list)
        if sample_list:  # Only add non-empty samples
            depadded_list.append(sample_list)
    return depadded_list

def pad(original_list, pad_to=100, padding_token=[-1, -1, -1, -1]):
    # Create a new list with the target length
    padded_list = original_list.copy()
    
    # Pad the list to the target length
    while len(padded_list) < pad_to:
        padded_list.append(padding_token)
    
    return padded_list

def depad(padded_tensor, padding_token=[-1, -1, -1, -1]):
    padding_token = torch.tensor(padding_token, dtype=padded_tensor.dtype)
    
    depadded_list = []
    for sample in padded_tensor:
        sample_list = []
        for row in sample:
            if not torch.equal(row, padding_token):
                sample_list.append(row.tolist())
        depadded_list.append(sample_list)
    
    return depadded_list

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

# prompt: "results" is a list of "batch_size" number of dicts. each dict has a key "boxes", whose value is a tensor of shape (100,4). so as you can probably tell, the value contains 100 bboxes of 4 coordinates each. the bboxes are of the format (top_left_x, top_left_y, bottom_right_x, bottom_right_y). i need to collect all of the "boxes" values into a tensor of shape (batch_size, 100, 4), and i must convert them to the format (top_left_x, top_left_y, width, height)
# Function to convert (x0, y0, x1, y1) to (x0, y0, width, height)
def convert_to_width_height(boxes):
    # Convert from (top_left_x, top_left_y, bottom_right_x, bottom_right_y) 
    # to (top_left_x, top_left_y, width, height)
    x0, y0, x1, y1 = boxes.unbind(-1)
    width = x1 - x0
    height = y1 - y0
    return torch.stack((x0, y0, width, height), dim=-1)

# prompt and unit test in gencode_testbed
def draw_bboxes_and_edges(image, prob_tensor, edge_tensor, bbox_thickness=2, line_thickness=2):
    # Define the color mapping for the classes
    colors = {
        1: (0, 0, 255),   # Red
        2: (255, 0, 0),   # Blue
        3: (0, 255, 0)    # Green
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