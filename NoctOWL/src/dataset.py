import cv2
import json
import os
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
import yaml
from PIL import Image
from src.util import get_processor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, OwlViTProcessor

def keep_only_rare(data):
    cat_ids = [cat['id'] for cat in data['categories'] if cat['frequency'] == 'r' or cat['frequency'] == 'f']
    anns = [ann for ann in data['annotations'] if ann['category_id'] in cat_ids]
    # assert sum([cat['instance_count'] for cat in data['categories'] if cat['frequency'] == 'r']) == len(anns)
    imm_ids = [ann['image_id'] for ann in anns]
    imms = [imm for imm in data['images'] if imm['id'] in imm_ids]
    cats = [cat for cat in data['categories'] if cat['id'] in cat_ids]
    
    # we make all the category id dense and starting from 0
    cat_ids_map = {cats[i]['id']: i for i in range(len(cats))}
    for cat in cats:
        cat['id'] = cat_ids_map[cat['id']]
    for ann in anns:
        ann['category_id'] = cat_ids_map[ann['category_id']]
    
    # add file_name to images:
    for imm in imms:
        imm['file_name'] = '/'.join(imm['coco_url'].split('/')[-2:])
        
    return {
        'annotations': anns,
        'categories': cats,
        'images': imms
    }

def get_images_dir(data_cfg):
    return data_cfg['images_path']

def remove_unprocessable_entries(data, training_cfg, perform_cleaning=False, keep_short_vocabularies=False):
    """
    Remove the annotations where the vocabulary has less then n_hardnegatives negatives and where the number of tokens is higher than the maximum for OWL
    """
    if not keep_short_vocabularies:
        data['annotations'] = [ann for ann in data['annotations'] if len(ann['neg_category_ids']) >= training_cfg['n_hardnegatives']]
    cats = {cat['id']: cat for cat in data['categories']}
    
    
    # we remove annotations where the vocabulary is too long for OWL
    processor = AutoProcessor.from_pretrained(training_cfg['base_model'])
    
    to_remove = []
    for ann in (data['annotations']):
        vocabulary = [cats[cat_id]['name'] for cat_id in [ann['category_id']] + ann['neg_category_ids'][:training_cfg['n_hardnegatives']]]
        
        len_vocabulary = processor(text=vocabulary, images=None, return_tensors="pt", padding=True)['input_ids'].shape[1]
        if len_vocabulary > 16:
            to_remove.append(ann['id'])
    
    # clean annotations  
    data['annotations'] = [ann for ann in data['annotations'] if ann['id'] not in to_remove]
    
    if perform_cleaning:
        # clean categories
        cat_ids = [cat_id for ann in data['annotations'] for cat_id in [ann['category_id']] + ann['neg_category_ids']]
        data['categories'] = [cat for cat in data['categories'] if cat['id'] in cat_ids]
        
        # clean images
        imm_ids = [ann['image_id'] for ann in data['annotations']]
        data['images'] = [imm for imm in data['images'] if imm['id'] in imm_ids]
        
    
    return data



class OwlDataset(Dataset):
    def __init__(self, image_processor, data_cfg, training_cfg, data_split='train'):
        self.images_dir = get_images_dir(data_cfg)
        self.image_processor = image_processor
        
        with open(data_cfg[f"{data_split}_annotations_file"]) as f:
            data = json.load(f)
            remove_unprocessable_entries(data, training_cfg)
            data = self.convert_to_train_format(data)
            # n_total = len(data)

        self.data = [{k: v} for k, v in data.items() if len(v)]

    def load_image(self, idx: int) -> Image.Image:
        url = list(self.data[idx].keys()).pop()
        path = os.path.join(self.images_dir, url.split('/')[-2],  url.split('/')[-1][:url.split('/')[-1].find(".jpg") + 4])
        image = Image.open(path).convert("RGB")
        # image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return image, path

    def load_target(self, idx: int):
        annotations = list(self.data[idx].values())

        # values results in a nested list
        assert len(annotations) == 1
        annotations = annotations.pop()

        labels = []
        boxes = []
        vocabularies = []
        for annotation in annotations:
            labels.append(annotation["label"])
            boxes.append(annotation["bbox"])
            vocabularies.append(annotation["vocabulary"]) 

        return labels, boxes, vocabularies

    def convert_to_train_format(self, data):
        new_data = {}
        
        images = {x['id']: x for x in data['images']}
        categories = {x['id']: x for x in data['categories']}
        count_vocabularies = {}
        
        for ann in data['annotations']:
            label = ann['category_id']
            vocabulary = [categories[id]['name'] for id in [label] + ann['neg_category_ids']]
            ann_obj = {
                'bbox': ann['bbox'],
                'label': label,
                'vocabulary': vocabulary
            }
            imm_path = images[ann['image_id']]['coco_url']
            # we use a little trick to simplify inferences:
            # since an image could have more group of objects with different vocabularies, we store an entry of the image one time for each vocabulary
            # in this way we have more inferences but the inference and the backpropagation is easier
            # to do it we append at the end of the imm_path a progressive id which indicates the number of group of objects (starting from 0) inside the image, and we remove it when an item of the dataloader is loaded
            count_vocabularies.setdefault(imm_path, {})
            count_vocabularies[imm_path].setdefault(label, max(count_vocabularies[imm_path].values()) + 1 if len(count_vocabularies[imm_path]) > 0 else 0)
            imm_path += str(count_vocabularies[imm_path][label])
            
            if imm_path in new_data:
                new_data[imm_path].append(ann_obj)
            else:
                new_data[imm_path] = [ann_obj]
                
        return new_data
            

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, path = self.load_image(idx)
        labels, boxes, vocabularies = self.load_target(idx)
        if isinstance(self.image_processor, OwlViTProcessor):
            w, h = image.size
        else:
            w = max(image.size)
            h = max(image.size)
        metadata = {
            "width": w,
            "height": h,
            "impath": path,
            "vocabularies": vocabularies
        }
        image = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        return image, torch.tensor(labels), torch.tensor(boxes), metadata
    
    
    
    
class LVISDataset(Dataset):
    def __init__(self, image_processor, data_cfg):
        self.images_dir = get_images_dir(data_cfg)
        self.image_processor = image_processor
        with open(data_cfg["lvis_annotations_file"]) as f:
            data = json.load(f)
            # data = keep_only_rare(data)
            data = self.convert_to_train_format(data)

        self.data = [{k: v} for k, v in data.items() if len(v)]
        self.max_anns = max(map(len, list(data.values())))

    def load_image(self, idx: int) -> Image.Image:
        url = list(self.data[idx].keys()).pop()
        path = os.path.join(self.images_dir, url)
        image = Image.open(path).convert("RGB")
        return image, path

    def load_target(self, idx: int):
        annotations = list(self.data[idx].values())
        # values results in a nested list
        assert len(annotations) == 1
        annotations = annotations.pop()

        labels = []
        boxes = []
        for annotation in annotations:
            labels.append(annotation["label"])
            boxes.append(annotation["bbox"])

        return labels, boxes
            
    def convert_to_train_format(self, data):
        new_data = {}
        
        images = {x['id']: x for x in data['images']}
        for ann in data['annotations']:
            key = '/'.join(images[ann['image_id']]['coco_url'].split('/')[-2:])
            entry = new_data.get(key, [])
            
            entry.append({
                'bbox': ann['bbox'],
                'label': ann['category_id'], 
            })
            new_data[key] = entry
        
        return new_data
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, path = self.load_image(idx)
        labels, boxes= self.load_target(idx)
        if isinstance(self.image_processor, OwlViTProcessor):
            w, h = image.size
        else:
            w = max(image.size)
            h = max(image.size)
        metadata = {
            "width": w,
            "height": h,
            "impath": path
        }
        image = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        padding_spaces = 0 # (self.max_anns - len(labels))
        labels = torch.tensor(labels + [-1] * padding_spaces)
        boxes = torch.tensor(boxes + [[-1, -1, -1, -1]] * padding_spaces)
        
        return image, labels, boxes, metadata


def get_dataloaders(
    data_cfg,
    training_cfg,
    num_workers=0
):
    lvis_evaluation = 'lvis_annotations_file' in data_cfg
    image_processor = get_processor(training_cfg['base_model'])

    train_dataset = OwlDataset(image_processor, data_cfg, training_cfg, 'train')
    test_dataset = OwlDataset(image_processor, data_cfg, training_cfg, 'test')
    lvis_dataset = LVISDataset(image_processor, data_cfg) if lvis_evaluation else None
        

    train_dataloader = DataLoader(train_dataset, batch_size=training_cfg['batch_size'], shuffle=True, num_workers=num_workers)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=training_cfg['batch_size'], shuffle=False, num_workers=num_workers)
    lvis_dataloader =  DataLoader(lvis_dataset, batch_size=1, shuffle=False, num_workers=0) if lvis_evaluation else None
    
    return train_dataloader, test_dataloader, lvis_dataloader