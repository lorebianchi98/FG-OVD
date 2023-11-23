import json
import os
from collections import defaultdict
from tqdm import tqdm
import copy
from utils.data_handler import are_attributes_included, remove_included_attributes, keeps_multi_attributes, remove_attributes_included, show_obj, intersection, intersection_multiple, check_equality, is_attribute_setted
PART_TO_REMOVE = ['inner_body', 'inner_side', 'inner_wall', 'shade_inner_side']


def factorize_attributes_inside_father(obj):
    """
    Factorize attributes inside father
    """
    colors = []
    materials = []
    transparencies = []
    patterns = []
    
    for part in obj['parts']:
        colors += [part['color']]
        materials += [part['material']]
        transparencies += [part['transparency']]
        patterns += [part['pattern_marking']]
        
    color_common = intersection_multiple(colors)
    material_common = intersection_multiple(materials)
    transparency_common = intersection_multiple(transparencies)
    pattern_common = intersection_multiple(patterns)
    
    if color_common != [] or material_common != [] or transparency_common != [] or pattern_common != []:
        obj['color'] += color_common
        obj['material'] += material_common
        obj['transparency'] += transparency_common
        obj['pattern_marking'] += pattern_common  
        
        new_parts = []
        for part in obj['parts']:
            part['color'] = list(set(part['color']) - set(color_common))
            part['material'] = list(set(part['material']) - set(material_common))
            part['transparency'] = list(set(part['transparency']) - set(transparency_common))
            part['pattern_marking'] = list(set(part['pattern_marking']) - set(pattern_common))
            if is_attribute_setted(part['color']) or is_attribute_setted(part['material']) or is_attribute_setted(part['transparency']) or is_attribute_setted(part['pattern_marking']):
                new_parts.append(part)
                
        if new_parts == []:
            del obj['parts']
        else:
            obj['parts'] = new_parts
        # we refactorize to check if we can factorize again
        if len(new_parts) > 1:
            factorize_attributes_inside_father(obj)
        
    return obj
    

def read_json(file_name):
    #Read JSON file
    with open(file_name) as infile:
        data = json.load(infile)
    return data

def write_json(data, file_name):
    # Write JSON file
    with open(file_name, "w") as outfile:
        json.dump(data, outfile)

def is_box_contained(box1, box2, margin=0):
    """
    Determine if box2 is contained within box1 with an additional margin
    box1: list of 4 coordinates [x1, y1, width, height]
    box2: list of 4 coordinates [x1, y1, width, height]
    margin: margin around box1 to allow box2 to overshoot
    """
    # changing the coordinates in the format: [x1, y1, x2, y2]
    box1[2] += box1[0]
    box1[3] += box1[1]
    box2[2] += box2[0]
    box2[3] += box2[1]
    
    assert box1[0] <= box1[2] and box1[1] <= box1[3], "Invalid box of the object"
    assert box2[0] <= box2[2] and box2[1] <= box2[3], "Invalid box of the part"
    if box2[0] >= box1[0]-margin and box2[1] >= box1[1]-margin and box2[2] <= box1[2]+margin and box2[3] <= box1[3]+margin:
        return True
    else:
        return False


import math

def get_center(box):
    x, y, w, h = box
    center_x = x + w/2
    center_y = y + h/2
    return center_x, center_y

def distance_between_boxes(box1, box2):
    center1 = get_center(box1)
    center2 = get_center(box2)
    dist = math.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)
    return dist

def min_distance(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    
    if x1 <= x2:
        if x1 + w1 >= x2:
            if y1 <= y2:
                if y1 + h1 >= y2:
                    # Boxes intersect
                    return 0
                else:
                    return y2 - (y1 + h1)
            elif y1 > y2 + h2:
                return y1 - (y2 + h2)
            else:
                # Boxes overlap vertically
                return 0
        else:
            if y1 <= y2:
                if y1 + h1 >= y2:
                    return x2 - (x1 + w1)
                elif y1 > y2 + h2:
                    return ((x2 - (x1 + w1))**2 + (y1 - (y2 + h2))**2)**0.5
                else:
                    return x2 - (x1 + w1)
            elif y1 > y2 + h2:
                return ((x2 - (x1 + w1))**2 + (y1 - (y2 + h2))**2)**0.5
            else:
                return x2 - (x1 + w1)
    else:
        if x2 + w2 >= x1:
            if y1 <= y2:
                if y1 + h1 >= y2:
                    return 0
                else:
                    return y2 - (y1 + h1)
            elif y1 > y2 + h2:
                return y1 - (y2 + h2)
            else:
                return 0
        else:
            if y1 <= y2:
                if y1 + h1 >= y2:
                    return x1 - (x2 + w2)
                elif y1 > y2 + h2:
                    return ((x1 - (x2 + w2))**2 + (y1 - (y2 + h2))**2)**0.5
                else:
                    return x1 - (x2 + w2)
            elif y1 > y2 + h2:
                return ((x1 - (x2 + w2))**2 + (y1 - (y2 + h2))**2)**0.5
            else:
                return x1 - (x2 + w2)

    

def merge_parts(data, filter_redundancy=True):
    """
    Merge parts with main objects
    if filter_redundancy=True,
                    remove_included_attributes(part, obj) it remove parts where the attributes are a subset of the main object
    """
    count_deleted_parts = 0 # counts the number of parts which are not added to any object
    count_added_parts = 0
    count_contained = 0
    count_no_parts = 0
    count_parts = 0
    total_parts = 0
    new_data = {}
    # iterating over all the images in the json file
    for img, objects in tqdm(data.items()):

        # we split objects and parts
        object_list = []
        part_list = []

        for item in objects:
            if item['supercategory'] == 'OBJECT':
                object_list.append(item)
            elif item['supercategory'] == 'PART':
                part_list.append(item)
            else:
                assert False, "Founded unexpected supercategory"
                
        # checking the distance from boxes
        for part in part_list:
            dist = [10000] * len(object_list)
            found = False
            for i, object in enumerate(object_list):
                if object['name'] == part['name'].split(':')[0]:
                    # dist[i] = min_distance(part['bbox'], object['bbox'])
                    dist[i] = distance_between_boxes(part['bbox'], object['bbox'])
                    # if the part is also completely contained in the box, we set the distance to -1,
                    # in order to prioritize, contained boxes
                    if is_box_contained(part['bbox'].copy(), object['bbox'].copy(), margin=10):
                        dist[i] = 0
                        count_contained += 1
                    found = True
            
                    
            if found:
                part['name'] = part['name'].split(':')[1] #removing the object category to the name of the part
                if part['name'] in PART_TO_REMOVE:
                    continue
                if 'parts' in object_list[dist.index(min(dist))]:
                    object_list[dist.index(min(dist))]['parts'].append(part)
                else:
                    object_list[dist.index(min(dist))]['parts'] = [part]
                count_added_parts += 1
            else:
                count_deleted_parts += 1
        
        # keeps_multi_attributes(object_list, add_multicolor=True) # aggregate multicolor attributes
        # filtering out the parts with object attributes redundant
        
        for obj in object_list:
            if 'parts' not in obj:
                count_no_parts += 1
                continue
            new_parts = []
            for part in obj['parts']:
                remove_attributes_included(part, obj) # removes attributes of the part repeated inside the object
                if not are_attributes_included(part, obj):
                    new_parts.append(part)
                    
            
            if new_parts != []:
                count_parts += 1
                obj['parts'] = new_parts
                if len(new_parts) > 1:
                    factorize_attributes_inside_father(obj)
            else:
                count_no_parts += 1
                del obj['parts']
                
                
        new_data[img] = object_list
    # print(f"Number of deleted parts: {count_deleted_parts}")
    # print(f"Number of added parts: {count_added_parts}")
    # print(f"Number of contained parts: {count_contained}\n")
    # print(f"Number of objects without parts: {count_no_parts}")
    # print(f"Number of objects with parts: {count_parts}")

    return new_data