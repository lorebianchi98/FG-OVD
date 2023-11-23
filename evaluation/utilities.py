DEVICE = 'cpu'

import torch
from torch import IntTensor, Tensor
import pickle, json

# READ/WRITE UTILITIES

def save_object(obj, path):
    """"Save an object using the pickle library on a file
    
    :param obj: undefined. Object to save
    :param fileName: str. Name of the file of the object to save
    """
    print("Saving " + path)
    with open(path, 'wb') as fid:
        pickle.dump(obj, fid)
        
def load_object(path):
    """"Load an object from a file
    
    :param fileName: str. Name of the file of the object to load
    :return: obj: undefined. Object loaded
    """
    try:
        with open(path, 'rb') as fid:
            obj = pickle.load(fid)
            return obj
    except IOError:
        return None   

def read_json(file_name):
    #Read JSON file
    with open(file_name) as infile:
        data = json.load(infile)
    return data

def write_json(data, file_name):
    # Write JSON file
    with open(file_name, "w") as outfile:
        json.dump(data, outfile)

# DATA HANDLING UTILITIES

def transform_predslist_to_dict(preds):
    result = {}
    for pred in preds:
        image = pred['image_filepath']
        if image not in result:
            result[image] = []
        result[image].append(pred)
    return result  

def get_image_ground_truth(data, image_id):
    """
    Given a dictionary 'data' and an 'image_id', returns a dictionary with 'boxes' and 'categories' information for
    that image.

    Args:
        data (dict): The data dictionary containing 'annotations'.
        image_id (int): The image_id for which to retrieve data.

    Returns:
        dict: A dictionary with 'boxes' and 'categories' information for the given image_id.
    """
    image_data = {'boxes': [], 'labels': [], 'annotation_id': []}  # Initialize the dictionary to store image data

    # Loop through each annotation in the 'annotations' list
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            # If the 'image_id' in the annotation matches the given 'image_id', append bbox and category_id to the lists
            image_data['annotation_id'].append(annotation['id']) 
            image_data['boxes'].append(annotation['bbox'])
            image_data['labels'].append(annotation['category_id'])

    image_data['boxes'] = convert_format(image_data['boxes'])
    # tensorize elements
    image_data['boxes'] = Tensor(image_data['boxes']).to(DEVICE)
    image_data['labels'] = IntTensor(image_data['labels']).to(DEVICE)
    
    return image_data

def calculate_iou(box1, box2):
    # Extract coordinates
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2
    
    # Calculate intersection coordinates
    x1_inter = max(x1_box1, x1_box2)
    y1_inter = max(y1_box1, y1_box2)
    x2_inter = min(x2_box1, x2_box2)
    y2_inter = min(y2_box1, y2_box2)
    
    # Calculate area of intersection
    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
    
    # Calculate area of both bounding boxes
    area_box1 = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)
    area_box2 = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1)
    
    # Calculate Union
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

def convert_format(boxes):
    for box in boxes:
        box[2] += box[0]
        box[3] += box[1]
    return boxes

def get_image_preds(preds, include_total_scores=True):
    labels = []
    scores = []
    boxes = []
    total_scores = []
    for pred in preds:
        labels += [x for x in pred['labels']]
        scores += [x for x in pred['scores']]
        boxes += ([x for x in pred['boxes']])
        if include_total_scores:
            total_scores += [x for x in pred['total_scores']]
        
    if type(boxes[0]) != torch.Tensor:
        return {
            'boxes': Tensor(boxes).to(DEVICE),
            'labels': IntTensor(labels).to(DEVICE),
            'scores': Tensor(scores).to(DEVICE),
            'total_scores': Tensor(total_scores).to(DEVICE),
            'category_id': preds[0]['category_id']
        }
    else:
        return {
            'boxes': torch.stack(boxes, dim=0).to(DEVICE),
            'labels': IntTensor(labels).to(DEVICE),
            'scores': Tensor(scores).to(DEVICE),
            'total_scores': Tensor(total_scores).to(DEVICE),
            'category_id': preds[0]['category_id']
        }