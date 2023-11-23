import random


from tqdm import tqdm
from utils.pickle_handler import loadObject
from utils.data_handler import intersection
import re


SEED = 123
random.seed(SEED)


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

def insert_color_placeholder(caption, color, index):
    caption = caption.replace('_', ' ')
    color = color.replace('_', ' ')
    
    #color found completely, with color composed by (shades + base color) or (basecolor)
    if caption.find(color) > -1:
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

def has_attributes_setted(obj):
    return (obj['color'] != [] and 'other' not in obj['color'][0]) or \
           (obj['material'] != [] and 'other' not in obj['material'][0]) or \
           (obj['pattern'] != [] and obj['pattern'][0] != 'plain' and 'other' not in obj['pattern'][0]) or \
           (obj['transparency'] != [] and obj['transparency'][0] != 'opaque' and 'other' not in obj['transparency'][0])

def create_hardnegatives_object(object, n_hardnegatives):
    """
    Creates a caption for an object and its part correlated with n_hardnegatives caption (if it is possible) 
    """
    COLORS = ['black', 'light_blue', 'blue', 'dark_blue', 'light_brown', 'brown', 'dark_brown', 'light_green', 'green', 'dark_green', 'light_grey', 'grey', 'dark_grey', 'light_orange', 'orange', 'dark_orange', 'light_pink', 'pink', 'dark_pink', 'light_purple', 'purple', 'dark_purple', 'light_red', 'red', 'dark_red', 'white', 'light_yellow', 'yellow', 'dark_yellow']
    MATERIALS = ['text', 'stone', 'wood', 'rattan', 'fabric', 'crochet', 'wool', 'leather', 'velvet', 'metal', 'paper', 'plastic', 'glass', 'ceramic']
    PATTERNS = ['plain', 'striped', 'dotted', 'checkered', 'woven', 'studded', 'perforated', 'floral']
    TRANSPARENCIES = ['opaque', 'translucent', 'transparent']
    
    
    objs = [object] if not 'parts' in object else [object] + object['parts']
    
    caption = object['answer']
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
        objs[i]['color_hn'] = [x for x in COLORS if x not in obj['color']] if (obj['color'] != [] and 'other' not in obj['color'][0]) else []
        objs[i]['material_hn'] = [x for x in MATERIALS if x not in obj['material']] if (obj['material'] != [] and 'other' not in obj['material'][0]) else []
        objs[i]['pattern_hn'] = [x for x in PATTERNS if x not in obj['pattern']] if (obj['pattern'] != [] and obj['pattern'][0] != 'plain' and 'other' not in obj['pattern'][0]) else []
        objs[i]['transparency_hn'] = [x for x in TRANSPARENCIES if x not in obj['transparency']] if (obj['transparency'] != [] and obj['transparency'][0] != 'opaque' and 'other' not in obj['transparency'][0]) else []
        
        # in order to make the negatives not nearly-impossible to guess, we remove all the shades the colors (for example: grey, light_grey, dark_grey)
        for color_obj in obj['color']:
            base_color = color_obj if len(color_obj.split('_')) == 1 else color_obj.split('_')[1]
            objs[i]['color_hn'] = [color for color in objs[i]['color_hn'] if base_color not in color]
            
            
    # if we are working with the main object, each hard negative array will be the intersection between all the array created for the object itself and parts
    for i in range(1, len(objs)):
        objs[0]['color_hn'] = intersection(objs[0]['color_hn'], objs[i]['color_hn']) if objs[i]['color_hn'] != [] else objs[0]['color_hn']
        objs[0]['material_hn'] = intersection(objs[0]['material_hn'], objs[i]['material_hn']) if objs[i]['material_hn'] != [] else objs[0]['material_hn']
        objs[0]['pattern_hn'] = intersection(objs[0]['pattern_hn'], objs[i]['pattern_hn']) if objs[i]['pattern_hn'] != [] else objs[0]['pattern_hn']
        objs[0]['transparency_hn'] = intersection(objs[0]['transparency_hn'], objs[i]['transparency_hn']) if objs[i]['transparency_hn'] != [] else objs[0]['transparency_hn']
        
        
    # we create the caption format  
    positive_caption = caption = clean_response(object['answer'].split('"')[1])
        
    for i, obj in enumerate(objs):
        colors = obj['color']
        materials = obj['material']
        patterns = obj['pattern']
        transparencies = obj['transparency']
        
        # old code which inverts only the first object
        # color = colors[0] if colors != [] and 'other' not in colors[0] else ""
        # material = materials[0] if materials != [] and 'other' not in materials[0] else ""
        # pattern = patterns[0] if patterns != [] and patterns[0] != 'plain' and 'other' not in patterns[0] else ""
        # transparency = transparencies[0] if transparencies != [] and transparencies[0] != 'opaque' and 'other' not in transparencies[0] else ""

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
        
    #creating hard negatives
    hardnegative_captions = []
    
    # print("\nPositive:\n%s\nNegatives" % positive_caption)
    
    for iter in range(0, n_hardnegatives):
        # if there are no others hard negatives to create we exit from the cycle
        if has_attributes == [False] * len(objs):
            break
        
        attributes = [true_colors.copy(), true_materials.copy(), true_patterns.copy(), true_transparencies.copy()] # matrix of attributes
        object_to_hn = random.choice([i for i, element in enumerate(has_attributes) if element]) # index of the object to retrieve hardnegative
        possible_hn = [objs[object_to_hn]['color_hn'], objs[object_to_hn]['material_hn'], objs[object_to_hn]['pattern_hn'], objs[object_to_hn]['transparency_hn']]
        
        #we create a list with all the possible attributes to invert
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
        assert new_hardnegative not in hardnegative_captions, "Hard negativo già creato"
        hardnegative_captions.append(new_hardnegative)
        
        # we remove the attribute from the list of possible hardnegatives
        possible_hn[attribute_to_hn].remove(attributes[attribute_to_hn][object_to_hn])
        # if there are no possible hardnegative for this object, we remove it from the list of object to invert
        has_attributes[object_to_hn] = possible_hn != [[],[],[],[]]
        
    assert len(hardnegative_captions) > 0, "zero hardnegatives generated, the caption is %s" % positive_caption  
    return positive_caption, hardnegative_captions

def create_hardnegatives(objects, n_hardnegatives):
    for object in tqdm(objects):
        positive, negatives = create_hardnegatives_object(object, n_hardnegatives)
        object['positive_caption'] = positive.replace('_', ' ')
        object['negative_captions'] = negatives
        
    return objects


def main():
    dataset_name = "paco_lvis_v1_test"
    data = loadObject('pickle/%s_captioned_correct1' % dataset_name)
    # creating hardnegatives
    data = create_hardnegatives(data, 10)
if __name__ == '__main__':
    main()