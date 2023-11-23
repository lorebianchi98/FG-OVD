import random


from tqdm import tqdm
import re, pickle, json
from copy import deepcopy


# **********************************************************
# UTILITIES 
# **********************************************************

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
        
def saveObject(obj, path):
    """"Save an object using the pickle library on a file
    
    :param obj: undefined. Object to save
    :param fileName: str. Name of the file of the object to save
    """
    print("Saving " + path + '.pkl')
    with open(path + ".pkl", 'wb') as fid:
        pickle.dump(obj, fid)
        
def loadObject(path):
    """"Load an object from a file
    
    :param fileName: str. Name of the file of the object to load
    :return: obj: undefined. Object loaded
    """
    try:
        with open(path + '.pkl', 'rb') as fid:
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

def intersection(array1, array2):
    set1 = set(array1)
    set2 = set(array2)
    return list(set1.intersection(set2))

def union(arr1, arr2):
    union_set = set(arr1 + arr2)
    union_array = list(union_set)
    return union_array


def group_images_per_id(images):
    result = {}
    for image in images:
        id = image['id']
        result[id] = image
    return result   

def get_paco_object(ann, file_path, paco_objects, show=True):
    """From the annotations it retrieves the objects PACO structure"""
    # file_path = file_path[5:] # remove 'coco/'
    for obj in paco_objects:
        if file_path == obj['image']:
            if obj['bbox'] == ann['bbox']:
                return obj
    # assert False, "Aiuto non ho trovato nulla"
    return None
    print("PACO structure not find!")


def show_captions(positive, negatives):
    print("Positive:\n%s\nNegatives:" % positive)
    for neg in negatives:
        print(neg)

def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else '' for i in text])

def clean_response(response):
    response = response.replace('\n', ' ')
    response = response.replace('\t', ' ')
    response = response.replace('  ', ' ')
    response = response.replace('<\s>', '')
    response = remove_non_ascii(response)
    return response

def simplify_attributes(object):
    obj = object.copy() # avoiding to modify original object
    
    for key, value in object.items():
        if key in ['color', 'material', 'pattern', 'transparency']:
            if value == []:
                obj.pop(key)
            else:
                obj[key] = value[0]
        if key == 'parts':
            obj['parts'][0] = object['parts'][0].copy()
            for key_part, value_part in object['parts'][0].items():
                if key_part in ['color', 'material', 'pattern', 'transparency']:
                    if value_part == []:
                        #obj['parts'][0].pop(key_part)
                        del obj['parts'][0][key_part]
                    else:
                        obj['parts'][0][key_part] = value_part[0]
    
    return obj

def check_color_in_caption(color, caption):
    regex = r'(?<!light )(?<!dark )' + re.escape(color)
    match = re.search(regex, caption, re.IGNORECASE)
    return bool(match)

def insert_color_placeholder(caption, color, index):
    caption = caption.replace('_', ' ')
    color = color.replace('_', ' ')
    
    #color found completely, with color composed by (shades + base color) or (basecolor)
    if check_color_in_caption(color, caption):
        caption = caption.replace(color, '{color[%d]}' % index, 1)# light grey
    # found only the base color
    elif len(color.split(' ')) > 1:
        new_capt = caption.replace(color.replace(' ', ''), '{color[%d]}' % index, 1) # lightgrey
        if new_capt == caption:
            caption = caption.replace(color.split(' ')[1], '{color[%d]}' % index, 1) # grey
        else:
            caption = new_capt
    
    return caption


def create_empty_fields(obj):
    obj['color'] = [obj['color']] if 'color' in obj else []
    obj['material'] = [obj['material']] if 'material' in obj else []
    obj['pattern'] = [obj['pattern']] if 'pattern' in obj else []
    obj['transparency'] = [obj['transparency']]if 'transparency' in obj else []
    
    return obj

def assert_categories(categories):
    # check that the position in the array of the category correspond to the id
    for i, cat in enumerate(categories):
        if 'id' in cat and cat['id'] != i:
            print("Element %d has id = %d" % (i, cat['id']))
            return False
        
    return True
        

def has_attributes_setted(obj):
    return (obj['color'] != [] and 'other' not in obj['color'][0]) or \
           (obj['material'] != [] and 'other' not in obj['material'][0]) or \
           (obj['pattern'] != [] and obj['pattern'][0] != 'plain' and 'other' not in obj['pattern'][0]) or \
           (obj['transparency'] != [] and obj['transparency'][0] != 'opaque' and 'other' not in obj['transparency'][0])
        
def fill_missing(categories):
    if not categories:
        return categories
    
    filled_categories = [None] * (max(category['id'] for category in categories) + 1)

    for category in categories:
        filled_categories[category['id']] = category

    return filled_categories

def delete_missing(categories):
    if not categories:
        return categories

    deleted_categories = []

    for category in categories:
        if 'id' in category:
            deleted_categories.append(category)

    return deleted_categories   
# ***************************************************************************************

def update_object(object, add_to_part=False):
    """
    Updates the object with the new attributes taken from the caption, new attributes will be added to the main object, and not to the parts
    """
    COLORS = ['black', 'light_blue', 'dark_blue', 'blue', 'light_brown', 'dark_brown', 'brown', 'light_green', 'dark_green', 'green', 'light_grey', 'dark_grey', 'grey', 'light_orange', 'dark_orange', 'orange', 'light_pink', 'dark_pink', 'pink', 'light_purple', 'dark_purple', 'purple', 'light_red', 'dark_red', 'red', 'white', 'light_yellow', 'dark_yellow', 'yellow',]
    MATERIALS = ['text', 'stone', 'wood', 'rattan', 'fabric', 'crochet', 'wool', 'leather', 'velvet', 'metal', 'paper', 'plastic', 'glass', 'ceramic']
    PATTERNS = ['plain', 'striped', 'dotted', 'checkered', 'woven', 'studded', 'perforated', 'floral', 'logo']
    TRANSPARENCIES = ['opaque', 'translucent', 'transparent']
    
    # creo un array per tipologia di attributo con tutti gli attributi presenti nell'oggetto e nelle sue parti
    objs = [object] + object['parts'] if 'parts' in object else [object]
    colors_act = []
    materials_act = []
    transparencies_act = []
    patterns_act = []
    
    caption = object['answer']
    if 'glass' in object['object']:
        caption = caption.replace('glass', '#####', 1)
    
    for obj in objs:
        colors_act = union(obj['color'], colors_act)
        materials_act = union(obj['material'], materials_act)
        transparencies_act = union(obj['transparency'], transparencies_act)
        patterns_act = union(obj['pattern'], patterns_act)
    
    # cerco tutti gli attributi provenienti da PACO presenti nella caption  
    colors_new = []
    materials_new = []
    transparencies_new = []
    patterns_new = []
    for color in COLORS:
        color_clean = color.replace('_', ' ')
        if caption.find(color_clean) > -1:
            caption = caption.replace(color_clean, '#')
            colors_new.append(color)
    for material in MATERIALS:
        material = material.replace('_', ' ')
        if caption.find(material) > -1:
            caption = caption.replace(material, '#')
            materials_new.append(material)
    for pattern in PATTERNS:
        pattern = pattern.replace('_', ' ')
        if caption.find(pattern) > -1:
            caption = caption.replace(pattern, '#')
            patterns_new.append(pattern)
    for transparency in TRANSPARENCIES:
        transparency = transparency.replace('_', ' ')
        if caption.find(transparency) > -1:
            caption = caption.replace(transparency, '#')
            transparencies_new.append(transparency)
      
    # cerco tra l'oggetto e le sue parti per quali attributi non ho trovato corrispondenza
    for obj in objs:
        obj['color'] = intersection(obj['color'], colors_new)
        obj['material'] = intersection(obj['material'], materials_new)
        obj['pattern'] = intersection(obj['pattern'], patterns_new)
        obj['transparency'] = intersection(obj['transparency'], transparencies_new)
    
    
    colors_new = [x for x in colors_new if x not in colors_act]
    materials_new = [x for x in materials_new if x not in materials_act]
    transparencies_new = [x for x in transparencies_new if x not in transparencies_act]
    patterns_new = [x for x in patterns_new if x not in patterns_act]
    
    # se per un attributo non ho trovato corrispondenza, lo aggiungo all'oggetto principale
    n_additions = max(len(colors_new), len(materials_new), len(transparencies_new), len(patterns_new)) - 1
    if not add_to_part or n_additions == 0:
        if colors_new != []:
            object['color'] += colors_new
        if materials_new != []:
            object['material'] += materials_new
        if transparencies_new != []:
            object['transparency'] += transparencies_new
        if patterns_new != []:
            object['pattern'] += patterns_new
    else:
        object['parts'] = [{
                    'name': 'not important',
                    'color': [],
                    'material': [],
                    'pattern': [],
                    'transparency': []
                }] * n_additions
        if colors_new != []:
            object['color'] += [colors_new[0]]
        if materials_new != []:
            object['material'] += [materials_new[0]]
        if transparencies_new != []:
            object['transparency'] += [transparencies_new[0]]
        if patterns_new != []:
            object['pattern'] += [patterns_new[0]]
            
        for i in range(1, n_additions + 1):
            if len(colors_new) > i:
                object['parts'][i - 1]['color'] += [colors_new[i]]
            if len(materials_new) > i:
                object['parts'][i - 1]['material'] += [materials_new[i]]
            if len(transparencies_new) > i:
                object['parts'][i - 1]['transparency'] += [transparencies_new[i]]
            if len(patterns_new) > i:
                object['parts'][i - 1]['pattern'] += [patterns_new[i]]
            
        
        
    return object
    

def create_hardnegatives_object(object, n_hardnegatives, hn_mask=[True, True, True, True], n_attributes_change=1):
    """
    Creates a caption for an object and its part correlated with n_hardnegatives caption (if it is possible) 
    hn_mask: array of boolean, if the value is True, the attribute will be considered for the hardnegative generation
    indexes of hn_mask:
    0: color
    1: materials
    2: pattern
    3: transparency
    """
    COLORS = ['black', 'light_blue', 'blue', 'dark_blue', 'light_brown', 'brown', 'dark_brown', 'light_green', 'green', 'dark_green', 'light_grey', 'grey', 'dark_grey', 'light_orange', 'orange', 'dark_orange', 'light_pink', 'pink', 'dark_pink', 'light_purple', 'purple', 'dark_purple', 'light_red', 'red', 'dark_red', 'white', 'light_yellow', 'yellow', 'dark_yellow']
    MATERIALS = ['text', 'stone', 'wood', 'rattan', 'fabric', 'crochet', 'wool', 'leather', 'velvet', 'metal', 'paper', 'plastic', 'glass', 'ceramic']
    PATTERNS = ['plain', 'striped', 'dotted', 'checkered', 'woven', 'studded', 'perforated', 'floral', 'logo']
    TRANSPARENCIES = ['opaque', 'translucent', 'transparent']
    
    
    
    AMBIGUITY_GROUPS = [
        ['dark_blue', 'dark_brown', 'dark_grey', 'black'], # near black
        ['white', 'light_grey', 'light_green', 'light_blue', 'light_brown'], # near white
        ['brown', 'grey'], # brown and grey
        ['light_orange', 'orange', 'light_yellow', 'yellow'], # near yellow
        ['light_orange', 'orange', 'light_red', 'red'], # near red
        ['orange', 'dark_orange', 'red', 'dark_red'], # near red
        ['orange', 'dark_orange', 'yellow', 'dark_yellow'], # near yellow
        ['light_pink', 'pink', 'light_purple', 'purple', 'light_red', 'red'], # near purple
        ['dark_pink', 'pink', 'dark_purple', 'purple', 'dark_red', 'red'], # near pink
    ]
    objs = [object] if not 'parts' in object else [object] + object['parts']
    
    has_attributes = [False] * len(objs) # element i check if obj[i] has attributes
    true_colors, true_materials, true_patterns, true_transparencies = [], [], [], []
    object_names = []
    for i in range(0, len(objs)):
        has_attributes[i] = has_attributes_setted(objs[i])
        # for each object we create a list with all the possible hardnegatives for each setted attribute
        # if an attribute is not setted, the list is empty
        # Exceptions: the following value for the attributes are considered as null:
        # color: other
        # material: other
        # pattern: plain, other
        # transparency: opaque, other
        obj = objs[i]
        # obj = create_empty_fields(obj)
        objs[i]['color_hn'] = [x for x in COLORS if x not in obj['color']] if (obj['color'] != [] and 'other' not in obj['color'][0] and hn_mask[0]) else []
        objs[i]['material_hn'] = [x for x in MATERIALS if x not in obj['material']] if (obj['material'] != [] and 'other' not in obj['material'][0] and hn_mask[1]) else []
        objs[i]['pattern_hn'] = [x for x in PATTERNS if x not in obj['pattern']] if (obj['pattern'] != [] and obj['pattern'][0] != 'plain' and 'other' not in obj['pattern'][0] and hn_mask[2]) else []
        objs[i]['transparency_hn'] = [x for x in TRANSPARENCIES if x not in obj['transparency']] if (obj['transparency'] != [] and obj['transparency'][0] != 'opaque' and 'other' not in obj['transparency'][0] and hn_mask[3]) else []
    
        # in order to make the negatives not nearly-impossible to guess, we remove all the shades the colors (for example: grey, light_grey, dark_grey)
        for color_obj in obj['color']:
            base_color = color_obj if len(color_obj.split('_')) == 1 else color_obj.split('_')[1]
            objs[i]['color_hn'] = [color for color in objs[i]['color_hn'] if base_color not in color]
            
        # in order to make the negatives not nearly-impossible to guess, we remove all the ambiguos color
        for color_obj in obj['color']:
            for group in AMBIGUITY_GROUPS:
                if color_obj in group:
                    for color in group:
                        objs[i]['color_hn'] = [color_hn for color_hn in objs[i]['color_hn'] if color_hn != color]
        
            
            
    # if we are working with the main object, each hard negative array will be the intersection between all the array created for the object itself and parts
    for i in range(1, len(objs)):
        objs[0]['color_hn'] = intersection(objs[0]['color_hn'], objs[i]['color_hn']) if objs[i]['color_hn'] != [] else objs[0]['color_hn']
        objs[0]['material_hn'] = intersection(objs[0]['material_hn'], objs[i]['material_hn']) if objs[i]['material_hn'] != [] else objs[0]['material_hn']
        objs[0]['pattern_hn'] = intersection(objs[0]['pattern_hn'], objs[i]['pattern_hn']) if objs[i]['pattern_hn'] != [] else objs[0]['pattern_hn']
        objs[0]['transparency_hn'] = intersection(objs[0]['transparency_hn'], objs[i]['transparency_hn']) if objs[i]['transparency_hn'] != [] else objs[0]['transparency_hn']
        
    # we create the caption format  
    positive_caption = caption = clean_response(object['answer'].split('"')[1])
    # we have to consider that an object glass could have the material glass, so we replace the object name to avoid problems
    if 'glass' in object['object']:
        caption = caption.replace('glass', '######', 1)
        
    for i, obj in enumerate(objs):
        colors = obj['color']
        materials = obj['material']
        patterns = obj['pattern']
        transparencies = obj['transparency']
        
        # Randomly select an element from the list, excluding the banned values
        color = random.choice([c for c in colors if 'other' not in c]) if colors else ""
        material = random.choice([m for m in materials if 'other' not in m]) if materials else ""
        pattern = random.choice([p for p in patterns if 'other' not in p and 'plain' not in p]) if patterns else ""
        transparency = random.choice([t for t in transparencies if 'other' not in t and 'opaque' not in t]) if transparencies else ""

        true_colors.append(color)
        true_materials.append(material)
        true_patterns.append(pattern)
        true_transparencies.append(transparency)
        
        # color placeholder
        if obj['color_hn'] != []:
            new_capt = insert_color_placeholder(caption, color, i)
            # no hard negative to insert
            if new_capt == caption:
                obj['color_hn'] = []
                true_colors[i] = ""
            else:
                caption = new_capt
        # material placeholder
        if obj['material_hn'] != []:
            # using regex we substitute the entire word (for example 'wooden' will be entirely substituted even if the material is 'wood')
            pattern_re = re.compile(r'\b\w*{}+\w*\b'.format(re.escape(material)), re.IGNORECASE)
            new_capt, num_subs = pattern_re.subn(r'{material[%d]}' % i, caption, count=1)

            replaced_word = pattern_re.findall(caption)[0] if num_subs > 0 else None
            true_materials[-1] = replaced_word
            # new_capt = caption.replace(material, '{material[%d]}' % i, 1)
            # no hard negative to insert
            if new_capt == caption:
                obj['material_hn'] = []
                true_materials[i] = ""
            else:
                caption = new_capt
        # pattern placeholder
        if obj['pattern_hn'] != []:
            new_capt = caption.replace(pattern, '{pattern[%d]}' % i, 1)
            # no hard negative to insert
            if new_capt == caption:
                obj['pattern_hn'] = []
                true_patterns[i] = ""
            else:
                caption = new_capt
        # transparency placeholder
        if obj['transparency_hn'] != []:
            new_capt = caption.replace(transparency, '{transparency[%d]}' % i, 1)
            # no hard negative to insert
            if new_capt == caption:
                obj['transparency_hn'] = []
                true_transparencies[i] = ""
            else:
                caption = new_capt
                
        # setting the has_attribute list
        has_attributes[i] = obj['color_hn'] != [] or obj['material_hn'] != [] or obj['transparency_hn'] != [] or obj['pattern_hn'] != []
    
    # remove eventually light and dark not eliminated
    if ' light {' in caption:
        caption = caption.replace(' light ', ' ')
        
    if 'dark {' in caption:
        caption = caption.replace(' dark ', ' ')
        
    hardnegative_captions = []
    
    # we have to consider that an object glass could have the material glass, so we replace the object name to avoid problems
    if 'glass' in object['object']:
        caption = caption.replace('######', 'glass')
        
    for iter in range(0, n_hardnegatives):
        # if there are no others hard negatives to create we exit from the cycle
        if has_attributes == [False] * len(objs):
            break
        
        attributes = [true_colors.copy(), true_materials.copy(), true_patterns.copy(), true_transparencies.copy()] # matrix of attributes
        object_to_hn = random.choice([i for i, element in enumerate(has_attributes) if element]) # index of the object to retrieve hardnegative
        possible_hn = [objs[object_to_hn]['color_hn'], objs[object_to_hn]['material_hn'], objs[object_to_hn]['pattern_hn'], objs[object_to_hn]['transparency_hn']]
        
        # we create a list with all the possible attributes to invert
        attributes_invertible = []
        attributes_invertible += [0] if possible_hn[0] != [] else []
        attributes_invertible += [1] if possible_hn[1] != [] else []
        attributes_invertible += [2] if possible_hn[2] != [] else []
        attributes_invertible += [3] if possible_hn[3] != [] else []
        assert attributes_invertible != [], "Non ci sono attributi da invertire, non dovrei essere qui"
        
        attribute_to_hn = random.choice(attributes_invertible) # choosing the attribute to change
        attributes[attribute_to_hn][object_to_hn] = random.choice(possible_hn[attribute_to_hn]) #choosing the new attribute for the hardnegative caption
        
        # we create the new hard negative caption
        new_hardnegative = caption.format(object=object_names,color=attributes[0],material=attributes[1],pattern=attributes[2],transparency=attributes[3])
        new_hardnegative = new_hardnegative.replace('_', ' ')
        # print(new_hardnegative)
        # if the number of attributes to change is greater than one, we create another hardnegative starting from the actual hardnegative
        if n_attributes_change > 1:
            no_hn = False
            already_generated = True
            # we generate a new hardnegative while the new_hardnegative is in the list of hardnegatives
            while already_generated: 
                new_obj = deepcopy(object)
                new_obj['answer'] = '"' + new_hardnegative + '"'
                new_hardnegative = create_hardnegatives_object(new_obj, 1, hn_mask, n_attributes_change-1)[1]
                # if we do not have a new_hardnegative, then we break from the cycle and return an empty list of hardnegatives
                if new_hardnegative == []:
                    no_hn = True
                    break
                else:
                    new_hardnegative = new_hardnegative[0]
                    already_generated = new_hardnegative in hardnegative_captions
                    
            # if we break from the cycle, we return no hardnegative
            if no_hn:
                return positive_caption, []
        assert new_hardnegative not in hardnegative_captions or n_attributes_change > 1, "Hard negativo già creato"
        hardnegative_captions.append(new_hardnegative)
        
        # we remove the attribute from the list of possible hardnegatives
        possible_hn[attribute_to_hn].remove(attributes[attribute_to_hn][object_to_hn])
        # if there are no possible hardnegative for this object, we remove it from the list of object to invert
        has_attributes[object_to_hn] = possible_hn != [[],[],[],[]]
    
    if all(hn_mask) and n_hardnegatives > 1 and n_attributes_change == 1:
        assert len(hardnegative_captions), "zero hardnegatives generated, the caption is %s" % positive_caption  
    return positive_caption, hardnegative_captions

def create_hardnegatives(objects, n_hardnegatives, hn_mask=[True, True, True, True]):
    for object in tqdm(objects):
        positive, negatives = create_hardnegatives_object(object, n_hardnegatives, hn_mask)
        object['positive_caption'] = positive.replace('_', ' ')
        object['negative_captions'] = negatives
        
    return objects

def create_hardnegatives_corrected_captions(data, paco_objects, n_hardnegatives, hn_mask=[True, True, True, True], n_attributes_change=1, verbose=False):
    COLORS = ['black', 'light_blue', 'blue', 'dark_blue', 'light_brown', 'brown', 'dark_brown', 'light_green', 'green', 'dark_green', 'light_grey', 'grey', 'dark_grey', 'light_orange', 'orange', 'dark_orange', 'light_pink', 'pink', 'dark_pink', 'light_purple', 'purple', 'dark_purple', 'light_red', 'red', 'dark_red', 'white', 'light_yellow', 'yellow', 'dark_yellow']
    MATERIALS = ['text', 'stone', 'wood', 'rattan', 'fabric', 'crochet', 'wool', 'leather', 'velvet', 'metal', 'paper', 'plastic', 'glass', 'ceramic']
    PATTERNS = ['plain', 'striped', 'dotted', 'checkered', 'woven', 'studded', 'perforated', 'floral', 'logo']
    TRANSPARENCIES = ['opaque', 'translucent', 'transparent']
    
    h = 0
    pos_negs_association = {} # associate to each id of a positive caption the ids of its negatives
    images = group_images_per_id(data['images'])
    # check that categories elements are correctly ordered
    print(f"{len(data['categories'])}")
    data['categories'] = fill_missing(data['categories'])
    to_del_ann = []
    assert_categories(data['categories'])
    for ann in tqdm(data['annotations']):
        # se la categoria è stata già processata, fillo semplicemente gli id dei negativi
        if ann['category_id'] in pos_negs_association:
            ann['neg_category_ids'] = pos_negs_association[ann['category_id']]
        else:
            # recupero la struttura PACO
            filepath = images[ann['image_id']]['file_name']
            add_parts = False
            object = get_paco_object(ann, filepath, paco_objects)
            if object is None:
                object = {
                    'object': 'not important',
                    'color': [],
                    'material': [],
                    'pattern': [],
                    'transparency': [],
                    'filled': False
                }
                add_parts = True
            # recupero l'oggetto categoria positiva
            pos_cat = data['categories'][ann['category_id']]
            # metto nel campo answer la caption corretta
            if 'proposed_caption' not in pos_cat:
                object['answer'] = '"' + pos_cat['name'] + '"'
            else:
                
                object['answer'] = '"' + pos_cat['proposed_caption'] + '"'
                del pos_cat['proposed_caption']
                object = update_object(object, add_to_part=add_parts)
            # creo i nuovi hard negatives
            positive, negatives = create_hardnegatives_object(object, n_hardnegatives, hn_mask, n_attributes_change=n_attributes_change)
            if verbose:
                show_captions(positive, negatives)
                print()
            # correggo gli oggetti categoria delle caption errate
            pos_cat['name'] = positive.replace('_', ' ')
            for i, neg_id in enumerate(ann['neg_category_ids'][:len(negatives)]):
                data['categories'][neg_id]['name'] = negatives[i]
            
            object['negative_captions'] = negatives
            
            # salvo la corrispondenza id_positive: id_negatives
            neg_ids = ann['neg_category_ids'][:len(negatives)]
            ann['neg_category_ids'] = ann['neg_category_ids'][:len(negatives)]
            pos_negs_association[ann['category_id']] = neg_ids
    
    for ann in to_del_ann:
        data['annotations'].remove(ann)
    print(f"Removed {len(to_del_ann)} annotations")
    data['categories'] = delete_missing(data['categories'])    
    return data

def main():
    data = read_json('merged.json')
    paco_objects = loadObject('datasets/not_captioned')
    # creating hardnegatives
    data = create_hardnegatives_corrected_captions(data, paco_objects, 10)
    write_json(data, 'merged_with_hardnegatives.json')
    
if __name__ == '__main__':
    main()