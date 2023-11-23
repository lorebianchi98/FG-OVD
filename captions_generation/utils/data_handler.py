"""
In this file are contained functions useful to manipulate the dataset.
"""
from PIL import Image
from torchvision.transforms import functional as func
from copy import deepcopy

# ****************************************************************************
# GET functions
# ****************************************************************************

ATTRIBUTES = ['color', 'material', 'pattern', 'pattern_marking', 'transparency']

def get_category_objects(anns, cat_per_id):
    """
    Get category objects from the annotation
    """
    cats = []
    for ann in anns:
        cats_id = [ann['category_id']] + ann['neg_category_ids']
        cats += [cat_per_id[id] for id in cats_id]
    
    return cats


def get_category_label(cat_id, data):
    for cat in data['categories']:
        if cat['id'] == cat_id:
            return cat['name']
        
    return None

def get_category(cat_id, data):
    for cat in data['categories']:
        if cat['id'] == cat_id:
            return cat
        
    return None

def get_image_path_from_id(imm_id, data):
    for imm in data['images']:
        if imm['id'] == imm_id:
            return imm['file_name']
    return None



# **************************************************************************
# UPDATE functions
# ****************************************************************************


def set_proposed_caption(cat_id, data, proposed_caption):
    cat = get_category(cat_id, data)
    assert cat is not None, "Category not found"
    cat['proposed_caption'] = proposed_caption
    return cat['proposed_caption']


def reduce_category_count(id, data):
    for cat in data['categories']:
        if cat['id'] == id:
            cat['instance_count'] -= 1
            assert cat['instance_count'] >= 0, "Reduced unused category"
            if cat['instance_count'] == 0:
                cat['image_count'] = 0
            return True
    return False

def delete_annotation(id, data):
    for ann in data['annotations']:
        if ann['id'] == id:
            res = reduce_category_count(ann['category_id'], data)
            assert res, 'Unexisting category'
            index = data['annotations'].index(ann)
            data['annotations'].pop(index)
            return True
    return False

def flag_annotation(id, data, value=None):
    """"
    Set the the needs_revision field to the opposite of the current value if value is None.
    """
    for ann in data['annotations']:
        if ann['id'] == id:
            ann['needs_revision'] = not ann['needs_revision'] if value is None else value
            return True

    return False

# *************************************************************************
# CONVERSION BOXES functions
# ****************************************************************************

def convert_absolute_to_relative_bounding_box(abs_bbox, sample):
    image = Image.open(sample.filepath)
    image = func.to_tensor(image).to('cpu')
    _, h, w = image.shape
    x1, y1, x2, y2 = abs_bbox
    return [x1 / w, y1 / h, x2 / w, y2 / h]

def convert_relative_to_absolute_bounding_box(relative_bbox, image_width, image_height):
    """
    Convert relative bounding box coordinates to absolute coordinates.

    Args:
        relative_bbox (list): List of relative bounding box coordinates [relative_x, relative_y, relative_width, relative_height].
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        list: List of absolute bounding box coordinates [x1, y1, x2, y2].
    """
    relative_x, relative_y, relative_width, relative_height = relative_bbox
    
    # Convert relative coordinates to absolute coordinates
    x1 = int(relative_x * image_width)
    y1 = int(relative_y * image_height)
    x2 = int((relative_x + relative_width) * image_width)
    y2 = int((relative_y + relative_height) * image_height)

    return [x1, y1, x2, y2]

# ****************************************************************************
# Other functions
# ****************************************************************************

def intersection(array1, array2):
    set1 = set(array1)
    set2 = set(array2)
    return list(set1.intersection(set2))

def intersection_multiple(list_of_lists):
    if not list_of_lists:
        return []

    # Convert the first list to a set
    result_set = set(list_of_lists[0])

    # Find the intersection with subsequent lists
    for i in range(1, len(list_of_lists)):
        result_set = result_set.intersection(list_of_lists[i])

    return list(result_set)


def count_obj_with_parts(data):
    n_obj_parts = 0
    n_obj_noparts = 0
    
    for obj in data:
        if 'parts' in obj and obj['parts'] != []:
            n_obj_parts += 1
        else:
            n_obj_noparts += 1
    print(f"Objects with parts: {n_obj_parts}")
    print(f"Objects without parts: {n_obj_noparts}")
    return n_obj_parts, n_obj_noparts


def keeps_multi_attributes(data, add_multicolor=False, keep_all=True):
    """
    keeps only objects which has attributes with more than 2 values
    """
    new_data = []
    for obj in data:
        multiattribute = False
        for key, values in obj.items():
            if key not in ['color','material', 'transparency', 'pattern', 'parts']:
                continue
            if key != 'parts':
                if len(values) == 1 and 'multi' in values[0]:
                    multiattribute = True
                if len(values) > 2:
                    obj[key] = ['multi-%s' % key] if add_multicolor else obj[key]
                    multiattribute = True
            else:
                for part in values:
                    for key_part, values_part in part.items():
                        if key_part not in ['color','material', 'transparency', 'pattern']:
                            continue
                        if len(values_part) == 1 and 'multi' in values_part[0]:
                            multiattribute = True
                        if len(values_part) > 2:
                            part[key_part] = ['multi-%s' % key_part] if add_multicolor else part[key_part]
                            multiattribute = True
        if multiattribute:
            new_data.append(obj)
    if keep_all:
        return data
    return new_data


def remove_strings_with_sequence(arr, sequence_list=['other', 'plain', 'opaque']):
    result = []
    for string in arr:
        contains_sequence = False
        for sequence in sequence_list:
            if sequence in string:
                contains_sequence = True
                break
        if not contains_sequence:
            result.append(string)
    return result

def are_attributes_included(obj_child, obj_parent):
    """
    Check if the attributes of the object obj1 are included in the object obj2
    """
    
    for key, values in obj_child.items():
        if key not in ATTRIBUTES:
            continue
        values = remove_strings_with_sequence(values)
        obj_parent[key] = remove_strings_with_sequence(obj_parent[key])
        # returning false if obj_child has an higher number of attributes than the parent
        if len(values) > len(obj_parent[key]):
            return False
        for value in values:
            if value not in obj_parent[key]:
                return False
    
    return True

def remove_attributes_included(obj1, obj2):
    """
    Removes the attributes of the object obj2 that are included in the object obj1
    """
    for key, values in obj1.items():
        if key not in ATTRIBUTES:
            continue
        values = remove_strings_with_sequence(values)
        obj2[key] = remove_strings_with_sequence(obj2[key])
        for value in values:
            if value in obj2[key]:
                obj2[key].remove(value)
    return obj2


def remove_included_attributes(obj_child, obj_parent):
    """
    Removes the attributes of the object obj1 that are included in the object obj2
    """
    for key, values in obj_child.items():
        if key not in ATTRIBUTES:
            continue
        values = remove_strings_with_sequence(values)
        obj_parent[key] = remove_strings_with_sequence(obj_parent[key])
        new_values = []
        for value in values:
            if value not in obj_parent[key]:
                new_values.append(value)
        obj_child[key] = new_values

def is_attribute_setted(attr):
    if len(attr) == 0:
        return False
    if 'other' in attr[0] or 'plain' in attr[0] or 'opaque' in attr[0]:
        return False
    return True

def check_equality(obj1, obj2):
    """
    Checks if two objects are equals.
    Returns:
    -1 if obj1 has less attributes than obj2 (obj1 attributes and parts describes obj2, but we can't say the inverse).
    0 if obj1 == obj2.
    1 otherwise
    """
    result = 0
    for key, value in obj1.items():
        if key not in ['name', 'color', 'material', 'transparency', 'pattern_marking']:
            continue
        # if obj1 has an attribute that obj2 does not have, we return 1
        if is_attribute_setted(obj1[key]) and not is_attribute_setted(obj2[key]):
            return 1
        # if obj2 has an attribute setted obj1 does not have, we set the result to -1, since objects are not equal and obj2 might be included in obj1 description 
        if not is_attribute_setted(obj1[key]) and is_attribute_setted(obj2[key]):
            result = -1
        # if both attributes are setted and they are different, we return 1
        if is_attribute_setted(obj1[key]) and is_attribute_setted(obj2[key]) and set(obj1[key]) != set(obj2[key]):
            return 1
    
    # we check if there is part similarity
    obj1_has_part = 'parts' in obj1 and len(obj1['parts']) > 0 
    obj2_has_part = 'parts' in obj2 and len(obj2['parts']) > 0 
    # if obj1 has parts and obj2 no, we return 1
    if obj1_has_part and not obj2_has_part:
        return 1
    # if obj2 has parts and obj1 no, we return -1, since objects are not equal and obj2 might be included in obj1 description 
    if not obj1_has_part and obj2_has_part:
        return -1
    # if neither obj1 nor obj2 has parts, we return the computation from attributes
    if not obj1_has_part and not obj2_has_part:
        return result
    # if both has parts, we check them
    if len(obj1['parts']) > len(obj2['parts']):
        return 1
    part_results = []
    parts_obj2 = deepcopy(obj2['parts'])
    for part1 in obj1['parts']:
        part_result = 0
        for index, part2 in enumerate(parts_obj2):
            part_result = check_equality(part1, part2)
            # parts included
            if part_result == -1:
                to_del = index
                break
            # parts equals
            if part_result == 0:
                to_del = index
                break
            # parts not comparable
            if part_result == 1:
                continue
        # this part is not comparable with no other part
        if part_result == 1:
            return 1
        # we remove the matched part
        parts_obj2.pop(to_del)
        # appending the result of the part
        part_results.append(part_result)
        
    if len(parts_obj2) == 0:
        # if the results is -1, we return -1
        if result == -1:
            return -1
        # if at least one part has less attributes, we return -1
        if -1 in part_results:
            return -1
        # if all the parts are equal to 0, and so the result, object are equals
        return 0
    # parts in obj2  left, obj1 is included in obj2 
    return -1

def show_obj(obj):
    attributes = ['object', 'name', 'color', 'material', 'transparency', 'pattern', 'pattern_marking']
    for key, value in obj.items():
        if key not in attributes:
            continue
        value = remove_strings_with_sequence(value) if key not in ['object', 'name'] else value # filtering uninteresting attributes
        if value != []:
            print(key, ': ', value)
    if 'parts' not in obj:
        return
    for i, part in enumerate(obj['parts']):
        print("Part %d:" % i)
        show_obj(part)

def keep_only_attributes(obj):
    attributes = ['object', 'name', 'color', 'material', 'transparency', 'pattern', 'parts']
    new_obj = {}
    for key, value in obj.items():
        if key in attributes:
            new_obj[key] = value
            
    return new_obj

def simplify_attributes(object):
    """Transform the attributes from lists to only one value"""
    obj = deepcopy(object) # avoiding to modify original object
    
    for key, value in object.items():
        if key in ['color', 'material', 'pattern', 'transparency']:
            if value == []:
                obj.pop(key)
            else:
                obj[key] = value
        else:
            obj.pop(key)      
        if key == 'parts':
            for i, part in enumerate(object['parts']):
                obj['parts'][i] = deepcopy(part)
                for key_part, value_part in object['parts'][i].items():
                    if key_part in ['color', 'material', 'pattern', 'transparency']:
                        if value_part == []:
                            del obj['parts'][i][key_part]
                        else:
                            obj['parts'][i][key_part] = value_part
                    else:
                        obj['parts'][i].pop(key)
    assert 'parts' not in obj or len(obj['parts']) == len(object['parts']), "Number of parts not congruent"
    
    return obj


def count_obj_with_parts_no_attr(objs):
    """Count the number of objects with parts and without attributes"""
    count = 0
    for obj in objs:
        if 'parts' in obj:
            for part in obj['parts']:
                if part['color'] == [] and part['material'] == [] and part['pattern'] == [] and part['transparency'] == []:
                    count += 1
                    break
    return count
            

def group_annotations_per_category(annotations):
    """Group annotations by categories"""
    result = {}
    for annotation in annotations:
        id = annotation['category_id']
        if id not in result:
            result[id] = []
        result[id].append(annotation)
    return result

def group_annotations_per_imagepath(annotations, data):
    """Group annotations by image filepath"""
    annotations = {}
    for annotation in data['annotations']:
        filepath = get_image_path_from_id(annotation['image_id'], data)
        if filepath not in annotations:
            annotations[filepath] = []
        annotations[filepath].append(annotation)
    return annotations

def assert_labels(detections, labels):
    """Check if the labels predicted are coherent with ground truths and negatives"""
    categories = []
    for detection in detections:
        categories += detection['neg_category_ids']
        # categories += [detection['neg_category_ids'][0] - 1] # ground truth label
        categories += detection['not_exhaustive_category_ids']
    
    is_valid = True
    for label in labels:
        if label not in categories:
            is_valid = False
            break
    
    return is_valid

def transform_predslist_to_dict_by_image(preds):
    """Group predictions by image path"""
    result = {}
    for pred in preds:
        image = pred['image_filepath']
        if image not in result:
            result[image] = []
        result[image].append(pred)
    return result  

def transform_predslist_to_dict_by_annotation(preds):
    """Group predictions by annotation id"""
    result = {}
    for pred in preds:
        ann_id = pred['annotation_id']
        if ann_id not in result:
            result[ann_id] = []
        result[ann_id].append(pred)
    return result  

def transform_annotations_list_to_dict(preds):
    result = {}
    for pred in preds:
        image = pred['image_id']
        if image not in result:
            result[image] = []
        result[image].append(pred)
    return result  

def transform_category_list_to_dict(cats):
    result = {}
    for cat in cats:
        id = cat['id']
        if id not in result:
            result[id] = []
        result[id] = cat
    return result  

    
def group_image_per_filepath(images):
    result = {}
    for image in images:
        id = image['file_name']
        if id not in result:
            result[id] = []
        result[id] = image
    return result  

def group_images_per_id(images):
    result = {}
    for image in images:
        id = image['id']
        result[id] = image
    return result  