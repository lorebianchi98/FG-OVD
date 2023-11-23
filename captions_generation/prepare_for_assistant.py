from utils.json_handler import read_json, write_json
from utils.pickle_handler import saveObject, loadObject
from utils.data_handler import simplify_attributes, keep_only_attributes, show_obj, remove_strings_with_sequence, are_attributes_included, check_equality
import random
from tqdm import tqdm
from copy import deepcopy

from itertools import groupby
SEED = 123


def group_objects_by_name(objects):
    sorted_objects = sorted(objects, key=lambda x: x['name'])
    grouped_objects = []

    for key, group in groupby(sorted_objects, key=lambda x: x['name']):
        grouped_objects.append(list(group))

    return grouped_objects

def simplify_colors(color_list):
    simplified_list = []
    color_counts = {}

    for color in color_list:
        parts = color.split('_')
        if len(parts) == 2:
            base_color = parts[1]
        else:
            base_color = color

        if color_counts.get(base_color, 0) == 0:
            simplified_list.append(color)
            color_counts[base_color] = 1

    return simplified_list

def has_attributes_setted(obj):
    if 'pattern_marking' in obj:
        return (obj['color'] != [] and 'other' not in obj['color'][0]) or \
            (obj['material'] != [] and 'other' not in obj['material'][0]) or \
            (obj['pattern_marking'] != [] and obj['pattern_marking'][0] != 'plain' and 'other' not in obj['pattern_marking'][0]) or \
            (obj['transparency'] != [] and obj['transparency'][0] != 'opaque' and 'other' not in obj['transparency'][0])
    else:
        return (obj['color'] != [] and 'other' not in obj['color'][0]) or \
            (obj['material'] != [] and 'other' not in obj['material'][0]) or \
            (obj['pattern'] != [] and obj['pattern'][0] != 'plain' and 'other' not in obj['pattern'][0]) or \
            (obj['transparency'] != [] and obj['transparency'][0] != 'opaque' and 'other' not in obj['transparency'][0])


def fill_objects(objects):
    grouped_objects = group_objects_by_name(objects)
    
    for group in grouped_objects:
        # we find the object filler
        obj_filler = {}
        actual_len = 0 # len parts of obj_filler
        for obj in group:
            if 'parts' in obj and actual_len < len(obj['parts']):
                obj_filler = obj
                actual_len = len(obj['parts'])
            elif 'parts' not in obj and actual_len == 0 and has_attribute_setted_global(obj) > -1:
                obj_filler = obj
                
        if obj_filler == {}:
            continue
        
        for obj in group:
            if obj == obj_filler:
                continue
            copy_obj(obj_filler, obj)
            obj['filled'] = True
            
    return
                
                
        
    
def refine_json(data, only_one_part=False, more_annotations_per_object=False, fills_subset=False):
    """
    If only one part is setted equal to True, an annotation for each part of the object is created
    if more_annotations_per_object is True, when an object is filled, if the original object has attributes, an annotation is created
    If fills_subset=True, an object is filled checking attributes, otherwise we take an object with attributes and fill all the otjer object with the same name in the image
    """
    new_data = []
    filled_data = []
    for img, objs_complete in tqdm(data.items()):
        # filling objects
        if not fills_subset:
            fill_objects(objs_complete)
        
        for obj_complete in objs_complete:
            
            
            # first we check if the object has attributes that needs to be filled from other objects in the image
            has_src_attr = has_attribute_setted_global(obj_complete)
            if has_src_attr == -1 and ('parts' in obj_complete or len(obj_complete) > 0):
                obj_complete['parts'] = []
            # ******** NEVER USED ********************************
            # check equalities if fills_subset
            if fills_subset:
                for obj_target in objs_complete:
                    # if the obj_complete is filled, we break the cycle
                    if 'filled' in obj_complete and obj_complete['filled']:
                        break
                    
                    # if the object is already filled, we skip it
                    if 'filled' in obj_target and obj_target['filled']:
                        continue
                    
                    # if the object is the same, we skip it
                    if obj_target['bbox'] == obj_complete['bbox'] and obj_target['segmentation'] == obj_complete['segmentation']:
                        continue
                    
                    # we are interested in do the filling only if the target has attributes
                    if has_attribute_setted_global(obj_target) == -1:
                        continue
                    
                    equality_res = check_equality(obj_complete, obj_target) 
                    new_obj = {}
                    # if we found an object more detailed, we creates a filled version of obj_complete
                    if equality_res == -1:
                        new_obj = copy_obj(obj_target, deepcopy(obj_complete))
                        new_obj['filled'] = True
                    # if object_complete can not be filled, we check if obj_target could be filled with obj_complete
                    # if True and more_annotations_per_object = True, we create an unfilled version of obj_complete
                    if more_annotations_per_object and equality_res == 1:
                        if check_equality(obj_target, obj_complete) == -1:
                            new_obj = copy_obj(obj_complete, deepcopy(obj_target))
                            new_obj['filled'] = True
                            
                    if new_obj != {}:
                        # if more_annotations_per_object = True, we append the copied object, otherwise we directly modify obj_complete
                        if more_annotations_per_object:
                            objs_complete.append(new_obj)
                        else:
                            copy_obj(new_obj, obj_complete)
                            obj_complete['filled'] = True
                
            # we want to check if the object has attributes
            if not has_attributes_setted(obj_complete):
                # if the object has no attribute we only keep the objects with attributes
                parts = obj_complete['parts'] if 'parts' in obj_complete else []
                obj_complete['parts'] = []
                for part in parts:
                    if has_attributes_setted(part):
                        obj_complete['parts'].append(part)
                        
                if len(obj_complete['parts']) == 0:
                    continue
                
                # if we are not in only_one_part mode, we keep only tha part with attributes, otherwise, since at least one attribute of the object is invertible, we keep all parts
                if not only_one_part:
                    obj_complete['parts'] = parts
            # all the parts are taken if the object has attributes
            else:
                parts = obj_complete['parts'] if 'parts' in obj_complete else []
            objs = [obj_complete] + obj_complete['parts'] if 'parts' in obj_complete else [obj_complete]
            new_objs = []
            
            for obj in objs:
                colors = obj['color']
                materials = obj['material']
                patterns = obj['pattern_marking']
                transparencies = obj['transparency']
                
                
                #assigning the true attributes
                color = colors[0] if colors != [] and 'other' not in colors[0] else ""
                material = materials[0] if materials != [] and 'other' not in materials[0] else ""
                pattern = patterns[0] if patterns != [] and patterns[0] != 'plain' and 'other' not in patterns[0] else ""
                transparency = transparencies[0] if transparencies != [] and transparencies[0] != 'opaque' and 'other' not in transparencies[0] else ""
                
                # we set the attributes of the object
                new_obj = {'object': obj['name']} if len(new_objs) == 0 else {'name': obj['name']}
                
                if color != '':
                    new_obj['color'] = remove_strings_with_sequence(simplify_colors(colors))
                else:
                    new_obj['color'] = []
                    
                if material != '':
                    new_obj['material'] = remove_strings_with_sequence(materials)
                else:
                    new_obj['material'] = []
                    
                if pattern != '':
                    new_obj['pattern'] = remove_strings_with_sequence(patterns)
                else:
                    new_obj['pattern'] = []
                    
                if transparency != '':
                    new_obj['transparency'] = remove_strings_with_sequence(transparencies)
                else:
                    new_obj['transparency'] = []
                    
                new_objs.append(new_obj)
            
            new_objs[0]['bbox'] = obj_complete['bbox']
            new_objs[0]['area'] = obj_complete['area']
            new_objs[0]['segmentation'] = obj_complete['segmentation']
            new_objs[0]['image'] = img
            new_objs[0]['filled'] = obj_complete['filled'] if 'filled' in obj_complete else False
            # merging main object with parts
            # we create an an object for each poobj1[key] != obj2[key]ssible part, in order to have multiple annotations
            if 'parts' in obj_complete:
                parts = []
                for part in new_objs[1:]:
                    # if only_one_part mode, we create an annotation for each part of the object
                    if only_one_part:
                        assert has_attributes_setted(new_objs[0]) or has_attributes_setted(part), "Neither the object and the part has attributes setted"
                        elem = new_objs[0].copy()
                        elem['parts'] = [part]
                        new_data.append(elem.copy())
                    else:
                        parts.append(part)
                if not only_one_part:
                    new_objs[0]['parts'] = parts
                    if new_objs[0]['filled'] == True:
                        filled_data.append(deepcopy(new_objs[0]))
                    else:
                        new_data.append(deepcopy(new_objs[0]))
            else:
                if new_objs[0]['filled'] == True:
                    filled_data.append(deepcopy(new_objs[0]))
                else:
                    new_data.append(deepcopy(new_objs[0]))
            
    return new_data, filled_data 



def check_how_many_equal(objs):
    """
    Checks how many objects without attributes have a counterpart with attributes
    """
    return None

def is_attribute_setted(attr):
    if len(attr) == 0:
        return False
    if 'other' in attr[0] or 'plain' in attr[0] or 'opaque' in attr[0]:
        return False
    return True

def has_attribute_setted_global(obj):
    """
    Returns:
    -1: the object has no attribute setted, nor a part of it
    0: the object has no attribute setted, but one part of it yes
    1: the object has attribute setted
    """
    if has_attributes_setted(obj):
        return 1
    
    if 'parts' not in obj or len(obj['parts']) == 0:
        return -1
    
    for part in obj['parts']:
        if has_attributes_setted(part):
            return 0
    
    return -1



def copy_obj(src, dest):
    """Copy attributes of src in dest"""
    for key, value in src.items():
        if key not in ['color', 'material', 'transparency', 'pattern_marking']:
            continue
        dest[key] = value
    
    if 'parts' not in  src or len(src['parts']) == 0:
        return dest
    
    dest['parts'] = []
    for part in src['parts']:
        part_dest = {}
        for key, value in part.items():
            if key not in ['name', 'color', 'material', 'transparency', 'pattern_marking']:
                continue
            part_dest[key] = value
        dest['parts'].append(part_dest)
        
    assert check_equality(src, dest) == 0, "Objects are different after the copy!"
    return dest

def create_simple_caption(obj):
    capt = 'a '
    for key, value in obj.items():
        if key not in ['color', 'material', 'transparency', 'pattern']:
            continue
        if value != []:
            capt += '%s ' % value[0]
    capt += '%s' % obj['name'] if 'name' in obj else obj['object']
    
    if not 'parts' in obj:
        return capt
    
    for part in obj['parts']:
        capt += ' with %s' % create_simple_caption(part)
    
    return capt 

def captionize(data):
    for obj in data:
         obj['answer'] = '"' + create_simple_caption(obj) + '"'
    return data


def main():
    data = loadObject("tmp/prepared")
    data = captionize(data)
if __name__ ==  '__main__':
    main()

