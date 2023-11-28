import argparse
import torch

from tqdm import tqdm

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
import torchvision.transforms as T

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random
from skimage import io as skimage_io

from utils.pickle_handler import saveObject, loadObject
from utils.json_handler import read_json, write_json

from pathlib import Path

from models import build_model
from main import get_args_parser

def load_image(image_path):
    # CORA receives images with CLIP normalization
    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]
    # load image
    image = cv2.imread(image_path)
    height = image.shape[0]
    width = image.shape[1]
    
    normalize = T.Compose([
        T.ToRGB(),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])

    transform = T.Compose(
        [
            normalize,
            T.RandomResize([800], max_size=1333),
        ]
    )
    image, _ = transform(image, None)  # 3, h, w
    return image, height, width

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


def evaluate_image(model, post_processors, img, vocabulary, original_width, original_height, MAX_PREDICTIONS=100):
    width, height = img.shape[-2:]
    # w, h = targets['orig_size']
    target_sizes = torch.tensor([[width, height]], device=device)
    img = img.to(device)
    img = img.unsqueeze(0)   # adding batch dimension
    outputs = model(img, categories=vocabulary)
    results = post_processors['bbox'](outputs, target_sizes)[0]
        
        
    # Get prediction logits
    logits = torch.max(outputs['logits'][0], dim=-1)
    all_scores = torch.sigmoid(outputs['logits'][0]).cpu().detach().numpy()
    
    # Get prediction labels and boundary boxes
    labels = results['labels']
    boxes = results['boxes'] 
    scores = results['scores']
        
    scores_filtered = []
    labels_filtered = []
    boxes_filtered = []
    total_scores_filtered = []
    
    boxes = [convert_to_x1y1x2y2(box, original_width, original_height) for box in boxes]
    # Combine the lists into tuples using zip
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to process')
    parser.add_argument('--out', type=str, required=True, help='Out path')
    parser.add_argument('--n_hardnegatives', type=int, default=10, help="Number of hardnegatives in each vocabulary")
    args = parser.parse_args()
    
    coco_path = '/home/lorenzobianchi/PacoDatasetHandling/coco/'
    if args.n_hardnegatives == 0 and '1_attributes' not in args.dataset:
        return
    
    # data = read_json('/home/lorenzobianchi/PacoDatasetHandling/jsons/captioned_%s.json' % dataset_name)
    data = read_json(args.dataset)
    
    model, criterion, post_processors = build_model(args)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_ema'])
        
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
        imm, w, h = load_image(image_filepath)
        output = evaluate_image(model, post_processors, imm, vocabulary, w, h)
        output['category_id'] = ann['category_id']
        output['vocabulary'] = vocabulary_id
        output['image_filepath'] = get_image_filepath(ann['image_id'], data['images'])
        output = adjust_out_id(output, vocabulary_id)
        complete_outputs.append(output)
        
    saveObject(complete_outputs, args.out)
if __name__ == '__main__':
    parser = argparse.ArgumentParser("CORA", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)