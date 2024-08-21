import yaml
import torch
from bs4 import BeautifulSoup as bs
from typing import List

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