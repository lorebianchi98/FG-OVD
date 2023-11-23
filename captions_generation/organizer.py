import json
import os
from collections import defaultdict
import numpy as np
from PIL import Image
import tqdm
import json

def write_json(data, file_name):
    # Write JSON file
    with open(file_name, "w") as outfile:
        json.dump(data, outfile)

def retrieve_name_from_ids(json, ids):
    """return a list of names given the ids and the dict to search in"""
    results = []
    for elem in json:
        if elem['id'] in ids:
            results.append(elem['name'])
    return results

def retrieve_elem_from_id(json, id):
    """return the dict given the ids and the dict to search in"""
    for elem in json:
        if elem['id'] == id:
            return elem

def test_image(id):
    # partendo da images
    imm = data['images'][id]
    imm_url = imm['flickr_url']
    neg_cat_ids = imm['neg_category_ids']
    cat_ids = imm['not_exhaustive_category_ids']

    neg_cat = retrieve_name_from_ids(data['categories'], neg_cat_ids)
    cat = retrieve_name_from_ids(data['categories'], cat_ids)



    #stampo l'url dell'immagine
    print(imm_url)
    print("Categories: %s" % cat)
    print("Negative categories: %s" % neg_cat)
    
def test_ann(id):
    #partendo da annotation
    ann = data['annotations'][id]
    
    cat_id = ann['category_id']
    imm_id = ann['image_id']
    bbox = ann['bbox']
    attr_ids = ann['attribute_ids']
    color_ids = ann['dom_color_ids']
    
    cat = retrieve_name_from_ids(data['categories'], [cat_id])
    imm = retrieve_elem_from_id(data['images'], imm_id)
    attr = retrieve_name_from_ids(data['attributes'], attr_ids)
    color = retrieve_name_from_ids(data['attributes'], color_ids)
    
    print("Immagine %d" % id)
    print(imm['flickr_url'])
    print("Categories: %s" % cat)
    print("Attributes: %s" % attr)
    print("Colors: %s" % color)
    print("Bounding boxes: %s\n" % bbox)
    
    
def get_supercategory(categories, id):
    for elem in categories:
        if elem['id'] == id:
            return elem['supercategory']
        
        
def filter_annotations(data):
    """Keeps only the annotations with attributes or that refer to parts"""
    
    filtered_ann = []
    
    for ann in data['annotations']:
        if len(ann['attribute_ids']) > 0:
            filtered_ann.append(ann)
        elif get_supercategory(data['categories'], ann['category_id']) == 'PART':
            filtered_ann.append(ann)
            
    return filtered_ann


def create_json_object_description(data):
    """
    Given the list with the annotation of the objects with attributes, we want to create a dict which describes
    the objects in each photo in the following format:
    image_path: {
        name: str,
        supercategory: str, 'OBJECT' or 'PART'
        bbox: list, #4 elements
        object_attribute: list of tuple, # name and type (color, material...)
        color: str,
        pattern-marking: str,
        material: str,
        transparency: str,
        object_part: {
            name: str,
            attribute: str, # or tuple?
        }
    }
    """
    annotations = data['annotations']
    types = ['color' for x in range(30)] + ['pattern_marking' for x in range(30,41)] + ['material' for x in range(41,55)] + ['transparency' for x in range(55,59)]
    objects = {}
    for ann in tqdm.tqdm(annotations):
        object = {}
        
        #retrieving ids of category, image, attributes and color + bbox
        cat_id = ann['category_id']
        imm_id = ann['image_id']
        attr_ids = ann['attribute_ids']
        color_ids = ann['dom_color_ids']
        
        # retrieving category, image, attributes and dominant color
        cat = retrieve_name_from_ids(data['categories'], [cat_id])
        imm = retrieve_elem_from_id(data['images'], imm_id)
        #attr = retrieve_name_from_ids(data['attributes'], attr_ids)
        color = retrieve_name_from_ids(data['attributes'], color_ids)
        
        #TODO: controllo se ci riferiamo ad una parte di un oggetto, nel caso si deve inserire la parte nel record relativo all'oggetto
        super_cat = retrieve_elem_from_id(data['categories'], cat_id)['supercategory']
        
        # recovering informations from attributes
        #attributes = [(retrieve_name_from_ids(data['attributes'], [attr_id])[0], types[attr_id]) for attr_id in attr_ids]
        
        attributes = {'color': color, 'pattern_marking': [], 'material': [], 'transparency': []}
        for attr_id in attr_ids:
            name = retrieve_name_from_ids(data['attributes'], [attr_id])[0]
            if name not in attributes[types[attr_id]]:
                attributes[types[attr_id]].append(name)
        
        # checks if important informations are missing:
        assert len(cat) > 0, "No category"
        assert imm is not None, 'No image path'
        #assert len(attr) > 0, "No attributes"
        
        
        object['name'] = cat[0]
        object['supercategory'] = super_cat
        object['bbox'] = ann['bbox']
        object['area'] = ann['area']
        object['segmentation'] = ann['segmentation']
        object['color'] = attributes['color']
        object['pattern_marking'] = attributes['pattern_marking']
        object['material'] = attributes['material']
        object['transparency'] = attributes['transparency']
        
        im_path = imm['file_name']
        if im_path in objects:
            objects[im_path].append(object)
        else:
            objects[im_path] = [object]
        
    return objects
        
def check_super_category(json, super_cats):
    # data['categories'] has ['OBJECT', 'PART']
    # data['attributes'] has ['ATTR']
    # data['part_categories'] has ['PART']
    
    for elem in json:
        assert elem['supercategory'] in super_cats, "identificata supercategoria %s" % elem['supercategory'] 
    


def count_image_url(annotations):
    print("Abbiamo da analizzare queste annotazioni: %d" % len(annotations))
    ids = []
    for ann in annotations:
        #retrieving ids of category, image, attributes and color + bbox
        cat_id = ann['category_id']
        imm_id = ann['image_id']
        if imm_id not in ids:
            ids.append(imm_id)
            
    return len(ids)


if __name__ == 'main':
    dataset_name = "paco_lvis_v1_train"
    #_PREDEFINED_PACO = '/home/lorenzobianchi/paco/annotations'
    # Derived parameters.
    dataset_file_name = dataset_name + '.json'
    image_root_dir = ''
    # Load dataset.
    with open(dataset_file_name) as f:
        data = json.load(f)
    
    # ann_w_attr = filter_annotations(data)
    objects = create_json_object_description(data['annotations'])
    write_json(objects, 'objects_per_image2.json')
