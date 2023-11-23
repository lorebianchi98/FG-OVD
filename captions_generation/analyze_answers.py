from open_assistant import visualize_conversations, remove_non_ascii, simplify_attributes
from question_remake import remove_ok_captions, reask_caption
from utils.pickle_handler import saveObject, loadObject
from prepare_for_assistant import has_attribute_setted_global 
from open_assistant import clean_response

import os
import torch
import argparse
import  json

from transformers import AutoTokenizer, AutoModelForCausalLM
import re

def contains_special_characters(string):
    pattern = r'[^A-Za-z.,\- _()\'"]'
    return bool(re.search(pattern, string))

NEW_PROMPTS = {
    0: "Your answer was too long. Create only one sentence for the object that describes what the object look like considering its attributes", # caption with more then 60 words
    1: "Your answer is a definition of what the object is. Give me a caption which only describe the object and its attributes", # ' is a ' inside the caption
    2: "You did not specified that you are describing a %s. Reformulate the caption with this addition", # object not inserted
    3: "You did not specify that the %s has a %s. Reformulate the caption with this addition", # part not inserted
    4: "Could you specify that the %s of the %s is %s?", # attribute not considered
    5: "Do not list the elements of the object. Summarize the description of the object in a natural language caption", # ':' in the caption
    6: "You gave me more than one captions. Summarize them in only one caption", # more than 2 '"'
    7: "Your answer contains a number not present in the JSON. Create a new caption considering only the attributes I gave you and without adding information", # a number in the caption
    8: "Answer is not complete. Write a complete caption", # only one '"' in the caption
    9: "Illegal characters in the caption. Remove them", # found illegal character
    10: "Ensure that the attributes are described using 'and' instead of 'or' to correctly represent all the specified attributes.", # 'or' in the caption
    11: "You used the word 'single', reformulate the caption without it" # 'single' in the caption
}

MOTIVATIONS = [
    "Too long answer", # 0
    "The sentence contains ' is a ', it is likely to be a vocabulary-like description", # 1
    "Object not inserted in the description", # 2
    "Part not utilized", # 3
    "Attributed not utilized", # 4
    "Contiene ':', probabilmente è una lista", # 5
    "Contiene troppe virgolette, probabilmente ha messo troppe caption", # 6
    "La caption data contiene numeri inventati", # 7
    "Caption not completed", # 8
    "Caption contains illegal characters", # 9
    "end answers", # 10
]

NAMES = [
    "too_long", # 0
    "descriptions", # 1
    "no_obj", # 2
    "no_part", # 3
    "no_attr", # 4
    "list", # 5
    "more_capt", # 6
    "numbers", # 7
    "uncomplete_caption", # 8
    "illegal_character", #9
    "ok" # 10
]  

def check_char_in_string(char_list, string):
    for char in char_list:
        if char in string:
            return True
    return False


# def are_attribute_considered(object):
#     obj = simplify_attributes(object)
#     resp = obj['answer']
#     parts = obj.get('parts') if 'parts' in obj else [{}]    
    
#     for key, value in obj.items():
#         if key not in ['color', 'material', 'transparency', 'pattern'] or value == []:
#             continue
#         value = value if len(value.split('_')) == 1 else value.split('_')[1] # if the value for example is 'light_brown', we search for 'brown'
#         if value not in resp:
#             return False, key, 'object', value
    
#     for part in parts:
#         for key, value in part.items():
#             if value == []:
#                 continue
#             if key == 'name':
#                 continue
#             value = value if len(value.split('_')) == 1 else value.split('_')[1] # if the value for example is 'light_brown', we search for 'brown'
#             if value not in resp:
#                 return False, key, 'part', value
    
#     return True, '', ''

def are_attribute_considered(object):
    obj = simplify_attributes(object)
    resp = obj['answer']
    parts = obj.get('parts') if 'parts' in obj else [{}]

    for key, values in obj.items():
        if key not in ['color', 'material', 'transparency', 'pattern'] or not values:
            continue

        for value in values:
            if 'multi' not in value:
                simplified_value = value.split('_')[-1]  # Extract the last part after splitting by '_'
                if simplified_value not in resp:
                    return False, key, obj['object'], simplified_value
            else:
                if value not in resp and value.replace('-', '') not in resp:
                    return False, key, obj['name'], simplified_value

    for part in parts:
        for key, values in part.items():
            if not values:
                continue
            if key == 'name':
                continue

            for value in values:
                if 'multi' not in value:
                    simplified_value = value.split('_')[-1]  # Extract the last part after splitting by '_'
                    if simplified_value not in resp:
                        return False, key, part['name'], simplified_value
                else:
                    if value not in resp and value.replace('-', '') not in resp:
                        return False, key, part['name'], simplified_value

    return True, '', ''



def has_part(obj):
    resp = obj['answer']
    
    #if the object has no parts, the caption could not have this error
    if 'parts' not in obj:
        return True 
    
    part_name = obj['parts'][0]['name'].split("_")[0]
    
    return part_name in resp

def has_object(obj):
    resp = obj['answer']
    
    obj_name = obj['object'].split("_")[0]
    
    return obj_name in resp


def count_new_prompts(objs, prompt_id):
    count = 0
    for obj in objs:
        if obj['new_prompt_id'] == prompt_id:
            count += 1
            
            
    return count

def filter_prompts(objs, prompt_id):
    new_objs = []
    for obj in objs:
        if obj['new_prompt_id'] == prompt_id:
            new_objs.append(obj)
    return new_objs

def evaluate_answers(objects):
    THRESH = 60 # max number of words allowed in the answer
    to_del = []
    
    
    for obj in objects:
        # if there are no attributes, the object is discarded
        if has_attribute_setted_global(obj) == -1:
            # print("Discarded object with caption %s" % obj['answer'])
            to_del.append(obj)
            continue
        resp = obj['answer']
        
        if len(resp.split(" ")) > THRESH:
            obj['new_prompt_id'] = 0
            obj['new_prompt'] = NEW_PROMPTS[obj['new_prompt_id']]
        elif ' is a ' in resp:
            obj['new_prompt_id'] = 1
            obj['new_prompt'] = NEW_PROMPTS[obj['new_prompt_id']]
        elif not has_object(obj):
            obj['new_prompt_id'] = 2
            obj['new_prompt'] = NEW_PROMPTS[obj['new_prompt_id']] % (obj['object'])
        elif not has_part(obj):
            obj['new_prompt_id'] = 3
            obj['new_prompt'] = NEW_PROMPTS[obj['new_prompt_id']] % (obj['object'], obj['parts'][0]['name'])
        elif not are_attribute_considered(obj)[0]:
            obj['new_prompt_id'] = 4
            _, attr_name, obj_type, attr_value = are_attribute_considered(obj)
            obj['new_prompt'] = NEW_PROMPTS[obj['new_prompt_id']] % (attr_name, obj_type, attr_value)
        elif ':' in resp:
            obj['new_prompt_id'] = 5
            obj['new_prompt'] = NEW_PROMPTS[obj['new_prompt_id']]
        elif resp.count('"') > 2:
            obj['new_prompt_id'] = 6
            obj['new_prompt'] = NEW_PROMPTS[obj['new_prompt_id']]
        elif check_char_in_string([str(num) for num in range(10)], resp):
            obj['new_prompt_id'] = 7
            obj['new_prompt'] = NEW_PROMPTS[obj['new_prompt_id']]
        elif resp.count('"') == 1:
            obj['new_prompt_id'] = 8
            obj['new_prompt'] = NEW_PROMPTS[obj['new_prompt_id']]
        elif contains_special_characters(clean_response(resp.split('"')[1])): # illegal character
            obj['new_prompt_id'] = 9
            obj['new_prompt'] = NEW_PROMPTS[obj['new_prompt_id']]
        elif ' or ' in resp:
            obj['new_prompt_id'] = 10
            obj['new_prompt'] = NEW_PROMPTS[obj['new_prompt_id']]
        elif 'single' in resp:
            obj['new_prompt_id'] = 11
            obj['new_prompt'] = NEW_PROMPTS[obj['new_prompt_id']]
        
        #it is ok
        else:
            obj['new_prompt_id'] = -1
            obj['new_prompt'] = ""
    
    # remove objects with no attributes to invert
    for obj in to_del:
        objects.remove(obj)
         
    return objects

def create_text_from_list(objs):
    text = ""
    for obj in objs:
        obj.pop("positive_caption")
        obj.pop("negative_captions")
        obj.pop("new_prompt")
        obj.pop("new_prompt_id")
        obj.pop("prompt")
        ans = obj.pop("answer")
        text += "%s\n%s\n\n" % (json.dumps(obj, indent=4), ans)
    return text

def save_categorized_answers(objs, path):    
    for i, obj in enumerate(objs):
        if len(obj) > 0:
            text = MOTIVATIONS[i] + '\n\n'
            text += "Count: %d\n\n" % len(obj)
            text += visualize_conversations(obj)
            with open(path + NAMES[i]  + '.txt', 'w') as f:
                f.write(remove_non_ascii(text))


def get_sum_splitted(objs_splitted):
    count = 0
    for objs in objs_splitted:
        count += len(objs)
        
    return count

def split_objects_by_question_to_ask(objects):
    N_TYPES = len(NAMES) - 1

    splitted_objs = []
    for i in range(N_TYPES):
        splitted_objs.append(filter_prompts(objects, i))
    splitted_objs.append(filter_prompts(objects, -1))   
    
    return splitted_objs

def correct_captions(objects, model, tokenizer, dataset_name, iterations=5, batch_size=16, ):
    ok_objs = []
    if iterations == 0:
        objects = evaluate_answers(objects) # adding the id of the next question to ask to each answer
        return remove_ok_captions(objects)[1]
    for i in range(iterations):
        objects = evaluate_answers(objects) # adding the id of the next question to ask to each answer
        objects, new_ok = remove_ok_captions(objects) # separating good answer to bad ones
        ok_objs += new_ok
        #splitted_objs = split_objects_by_question_to_ask(objects)[:-1]
        #save_categorized_answers(splitted_objs + [ok_objs], 'script/answer_analysis%d/' % i)

        print("Reasking the captions, step %d\n" % i)
        objects = reask_caption(model, tokenizer, objects, batch_size=batch_size) # reasking the caption to OpenAssistant
        
    # splitting answers before saving them        
    objects = evaluate_answers(objects) # adding the id of the next question to ask to each answer 
    objects, new_ok = remove_ok_captions(objects) # separating good answer to bad ones
    ok_objs += new_ok
    saveObject(ok_objs, 'pickle/%s_corrected_%s'  % (dataset_name, str(i)))
    saveObject(objects, 'pickle/%s_wrong_%s'  % (dataset_name, str(i)))
    
    return ok_objs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2, help='GPU number to use')
    parser.add_argument('--iterations', type=int, default=5, help='Number to iteration of questions to ask to OpenAssistant')
    args = parser.parse_args()
    
    
    iterations = args.iterations
    
    torch.cuda.set_device(args.gpu)
    SEED = 123
    torch.manual_seed(SEED) # setting fixed seed to make experiments reproducible
    
    os.environ['TRANSFORMERS_CACHE'] = 'cache/'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("tokenizers/open_assistant", padding_side="left")
    try:
        model = loadObject("models/open_assistant/model")
        model.cuda()
    except Exception as e:
        model = AutoModelForCausalLM.from_pretrained("models/open_assistant")
        model.half().cuda()
    model.cuda()
    print("Done!")
    
    
    
    #objects = loadObject('responses/captioned_objs_short')
    #objects = loadObject('objs_reasked')
    #objects = evaluate_answers(objects)
    #saveObject(objects, "script/reprompted_object")
    
    objects = loadObject('responses/captioned_objs_short')
    
    correct_captions(objects, model, tokenizer, "prappo", batch_size=16)
    
    
if __name__ == '__main__':
    main()