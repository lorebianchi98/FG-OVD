import argparse
import os
import sys

os.environ['TRANSFORMERS_CACHE'] = 'cache/'

from tqdm import tqdm
from utils.pickle_handler import save_object, load_object
from utils.json_handler import read_json, write_json

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from torchvision.ops import batched_nms

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

def get_category_name(id, categories):
    for category in categories:
        if id == category['id']:
            return category['name']
        
def get_image_filepath(id, images):
    for image in images:
        if id == image['id']:
            return image['file_name']

def adjust_out_id(output, vocabulary_id):
    for i in range(len(output['labels'])):
        output['labels'][i] = vocabulary_id[output['labels'][i]]
    return output


def create_vocabulary(ann, categories):
    vocabulary_id = [ann['category_id']] + ann['neg_category_ids']
    vocabulary_uncleaned = [get_category_name(id, categories) for id in vocabulary_id]
    
    vocabulary = []
    for voc in vocabulary_uncleaned:
        # voc = voc[:-1] if voc[-1] == '.' else voc # removing last '.'
        # voc = voc.replace('.', ',') # replace eventual ','
        if not voc.endswith("."):
            voc = voc + "."
        vocabulary.append(voc)
    
    return vocabulary, vocabulary_id

def sort_boxes_by_score(boxes, labels, scores, max_elem):
    """
    Sorts the boxes, labels, and scores arrays by the score in descending order.

    Args:
        boxes (list): List of boxes.
        labels (list): List of labels.
        scores (list): List of scores.

    Returns:
        dict: A dictionary containing the sorted boxes, labels, and scores arrays.
    """
    # Sort the boxes, labels, and scores arrays by score in descending order
    sorted_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    sorted_boxes = [boxes[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]

    # Create a dictionary to store the sorted arrays
    sorted_dict = {
        'boxes': sorted_boxes[:max_elem],
        'labels': sorted_labels[:max_elem],
        'scores': sorted_scores[:max_elem]
    }

    return sorted_dict



def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def apply_NMS(boxes, scores, labels, iou=0.5):
    indexes_to_keep = batched_nms(torch.stack(boxes, dim=0),
                       torch.FloatTensor(scores),
                       torch.IntTensor([0] * len(boxes)),
                       iou)
    
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    for x in indexes_to_keep:
        filtered_boxes.append(boxes[x])
        filtered_scores.append(scores[x])
        filtered_labels.append(labels[x])
        
    return filtered_boxes, filtered_scores, filtered_labels


def get_grounding_output(model, image, captions, w, h, max_elem=100, iou=0.5, box_threshold=0, cpu_only=False):
    # text_threshold = 0
    device = "cuda" if not cpu_only else "cpu"
    image = image.to(device)
    
    # we need to batch the inputs, in order to make a query for each caption
    images = image[None].repeat(len(captions), 1, 1, 1)
    
    with torch.no_grad():
        outputs = model(images, captions=captions)
    
    pred_boxes = []
    pred_scores = []
    pred_label_ids = []
    
    logits_batched = [x.cpu().sigmoid() for x in outputs['pred_logits']]
    boxes_batched = [x.cpu() for x in outputs['pred_boxes']]
    for logits, boxes, label_id in zip(logits_batched, boxes_batched, range(len(captions))):
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]
        
        for logit, box in zip(logits_filt, boxes_filt):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([w, h, w, h])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            pred_boxes.append(box)
            pred_label_ids.append(label_id)
            pred_scores.append(logit.max().item())

    pred_boxes, pred_scores, pred_label_ids = apply_NMS(pred_boxes, pred_scores, pred_label_ids, iou)[:max_elem]


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path of the json to process')
    parser.add_argument("--imgs_path", type=str, required=True, help="Path of the images to process")
    parser.add_argument('--out', type=str, required=True, help='Name of the dataset to process')
    args = parser.parse_args()

    # load dataset
    data = read_json(args.dataset_path)

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)
    device = "cuda" if not args.cpu_only else "cpu"
    model = model.to(device)
 
    complete_outputs = []
    for ann in tqdm(data['annotations']):
        image_filepath = args.imgs_path + '/' + get_image_filepath(ann['image_id'], data['images'])
        # load image
        image_pil, imm = load_image(image_filepath)
        vocabulary, vocabulary_id = create_vocabulary(ann, data['categories'])
        output = get_grounding_output(model, imm, vocabulary, image_pil.size[0], image_pil.size[1])
        if output == None:
            continue
        output['annotation_id'] = ann['id']
        output['image_filepath'] = get_image_filepath(ann['image_id'], data['images'])
        output = adjust_out_id(output, vocabulary_id)
        complete_outputs.append(output)
    
    save_object(complete_outputs, "backup")    
    save_object(complete_outputs, args.out)
    