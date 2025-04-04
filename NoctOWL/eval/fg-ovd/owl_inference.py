import argparse
import torch

from tqdm import tqdm

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from torchvision.ops import batched_nms

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random
import pickle

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import Owlv2Processor, Owlv2ForObjectDetection

SCORE_THRESH = 0.1
import numpy as np

def save_object(obj, path):
    # Ensure all directories along the path exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    print("Saving " + path)
    with open(path, 'wb') as fid:
        pickle.dump(obj, fid)

def read_json(file_name):
    with open(file_name) as infile:
        data = json.load(infile)
    return data

def convert_to_x1y1x2y2(bbox, img_width, img_height):
    """
    Convert bounding boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format.

    Args:
        bbox (np.array): NumPy array of bounding boxes in the format [cx, cy, w, h].
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        np.array: NumPy array of bounding boxes in the format [x1, y1, x2, y2].
    """
    cx, cy, w, h = bbox
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return np.array([x1, y1, x2, y2])

def apply_NMS(boxes, scores, labels, total_scores, iou=0.5):
    indexes_to_keep = batched_nms(torch.stack([torch.FloatTensor(box) for box in boxes], dim=0),
                       torch.FloatTensor(scores),
                       torch.IntTensor(labels),
                       iou)
    
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    filtered_total_scores = []
    deleted_boxes = []
    deleted_scores = []
    deleted_labels = []
    deleted_total_scores = []
    
    for x in range(len(boxes)):
        if x in indexes_to_keep:
            filtered_boxes.append(boxes[x])
            filtered_scores.append(scores[x])
            filtered_labels.append(labels[x])
            filtered_total_scores.append(total_scores[x])
        else:
            deleted_boxes.append(boxes[x])
            deleted_scores.append(scores[x])
            deleted_labels.append(labels[x])
            deleted_total_scores.append(total_scores[x])
    
    return filtered_boxes, filtered_scores, filtered_labels, filtered_total_scores



skipped_categories = 0
def evaluate_image(model, processor, im, vocabulary, MAX_PREDICTIONS=100, nms=False, v2=False):
    global skipped_categories
    # preparing the inputs
    inputs = processor(text=vocabulary, images=im, return_tensors="pt", padding=True).to(device)
    
    # if the tokens length is above 16, the model can't handle them
    if  inputs['input_ids'].shape[1] > 16:
        skipped_categories += 1
        return None
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        
        
    # Get prediction logits
    logits = torch.max(outputs['logits'][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()
    all_scores = torch.sigmoid(outputs['logits'][0]).cpu().detach().numpy()
    
    # Get prediction labels and boundary boxes
    labels = logits.indices.cpu().detach().numpy()
    boxes = outputs['pred_boxes'][0].cpu().detach().numpy()    
        
    scores_filtered = []
    labels_filtered = []
    boxes_filtered = []
    total_scores_filtered = []
    if not v2:
        height = im.shape[0]
        width = im.shape[1]
    else:
        height = max(im.shape)
        width = max(im.shape)
    
    boxes = [convert_to_x1y1x2y2(box, width, height) for box in boxes]
    # Combine the lists into tuples using zip
    if nms:
        # apply NMS
        boxes, scores, labels, all_scores = apply_NMS(boxes, scores, labels, all_scores)
    data = list(zip(scores, boxes, labels, all_scores))

    # Sort the combined data based on the first element of each tuple (score) in decreasing order
    sorted_data = sorted(data, key=lambda x: x[0], reverse=True)
    
    
    # filtering the predictions with low confidence
    for score, box, label, total_scores in sorted_data[:MAX_PREDICTIONS]:
        scores_filtered.append(score)
        labels_filtered.append(label)
        # boxes_filtered.append(convert_to_x1y1x2y2(box, width, height))
        boxes_filtered.append(box)
        total_scores_filtered.append(total_scores)
    
    return {
        'scores': scores_filtered,
        'labels': labels_filtered,
        'boxes': boxes_filtered,
        'total_scores': total_scores_filtered
    }
        
    

def get_category_name(id, categories):
    for category in categories:
        if id == category['id']:
            return category['name']
        
def get_image_filepath(id, images):
    for image in images:
        if id == image['id']:
            return image['file_name']

def create_vocabulary(ann, categories):
    vocabulary_id = [ann['category_id']] + ann['neg_category_ids']
    vocabulary = [get_category_name(id, categories) for id in vocabulary_id]
    
    return vocabulary, vocabulary_id

def adjust_out_id(output, vocabulary_id):
    for i in range(len(output['labels'])):
        output['labels'][i] = vocabulary_id[output['labels'][i]]
    return output

def remove_unprocessable_entries(data, n_hardnegatives=10, perform_cleaning=False):
    """
    Remove the annotations where the vocabulary has less then n_hardnegatives negatives and where the number of tokens is higher than the maximum for OWL
    """
    data['annotations'] = [ann for ann in data['annotations'] if len(ann['neg_category_ids']) >= n_hardnegatives]
    
    cats = {cat['id']: cat for cat in data['categories']}
    
    
    # we remove annotations where the vocabulary is too long for OWL
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch16')
    
    to_remove = []
    print("Searching unprocessable annotations...")
    for ann in tqdm(data['annotations']):
        vocabulary = [cats[cat_id]['name'] for cat_id in [ann['category_id']] + ann['neg_category_ids'][:n_hardnegatives]]
        len_vocabulary = processor(text=vocabulary, images=None, return_tensors="pt", padding=True)['input_ids'].shape[1]
        if len_vocabulary > 16:
            to_remove.append(ann['id'])
    
    # clean annotations  
    print("Cleaning annotations...")      
    data['annotations'] = [ann for ann in tqdm(data['annotations']) if ann['id'] not in to_remove]
    
    if perform_cleaning:
        # clean categories
        print("Cleaning categories...")
        cat_ids = [cat_id for ann in data['annotations'] for cat_id in [ann['category_id']] + ann['neg_category_ids']]
        data['categories'] = [cat for cat in tqdm(data['categories']) if cat['id'] in cat_ids]
        
        # clean images
        print("Cleaning images...")      
        imm_ids = [ann['image_id'] for ann in data['annotations']]
        data['images'] = [imm for imm in tqdm(data['images']) if imm['id'] in imm_ids]
        
    
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to process')
    parser.add_argument('--out', type=str, required=True, help='Out path')
    parser.add_argument('--model', type=str, default="google/owlvit-base-patch16", help='OWL configuration')
    parser.add_argument('--tokenizer', type=str, default="google/owlvit-base-patch16", help='OWL tokenizer configuration')
    parser.add_argument('--n_hardnegatives', type=int, default=5, help="Number of hardnegatives in each vocabulary")
    args = parser.parse_args()
    global skipped_categories
    
    coco_path = '../../coco/'
    # if args.n_hardnegatives == 0 and '1_attributes' not in args.dataset:
    #     return
    
    data = read_json(args.dataset)
    data = remove_unprocessable_entries(data, args.n_hardnegatives, True)
    if 'v1' in args.model:
        model = OwlViTForObjectDetection.from_pretrained(args.model)
        processor = OwlViTProcessor.from_pretrained(args.tokenizer)
    else:
        model = Owlv2ForObjectDetection.from_pretrained(args.model)
        processor = Owlv2Processor.from_pretrained(args.tokenizer)
    print(f"{args.model} model loaded")
    model = model.to(device)
    model.eval()
    
    complete_outputs = []
    categories_done = []
    for ann in tqdm(data['annotations']):
        # if the category is not done, we add it to the list
        if ann['category_id'] not in categories_done:
            categories_done.append(ann['category_id'])
        else:
            continue
        vocabulary, vocabulary_id = create_vocabulary(ann, data['categories'])
        # check if a number of hardnegatives is setted to non-default values
        # if it is, the vocabulary is clipped and if it is too short, we skip that image
        len_vocabulary = args.n_hardnegatives + 1
        if len(vocabulary) < len_vocabulary:
            continue
        vocabulary = vocabulary[:len_vocabulary]
        vocabulary_id = vocabulary_id[:len_vocabulary]
        image_filepath = coco_path + get_image_filepath(ann['image_id'], data['images'])
        imm = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)
        output = evaluate_image(model, processor, imm, vocabulary, nms=False, v2='v2' in args.model)
        if output == None:
            continue
        output['category_id'] = ann['category_id']
        output['vocabulary'] = vocabulary_id
        output['image_filepath'] = get_image_filepath(ann['image_id'], data['images'])
        output = adjust_out_id(output, vocabulary_id)
        complete_outputs.append(output)
        
    save_object(complete_outputs, args.out)
    print("Skipped categories: %d/%d" % (skipped_categories, len(categories_done)))
if __name__ == '__main__':
    main()