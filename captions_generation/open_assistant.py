SEED = 123

import os, argparse
os.environ['TRANSFORMERS_CACHE'] = 'cache/'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.pickle_handler import saveObject, loadObject

from tqdm import tqdm
import json

import copy


def create_context(obj):
    question = """
I will give you a json describing an object. I want you to give me a one natural language caption which describes the object. These are four examples: 
1:
{
    "object": "scissors",
    "color": ["blue"],
    "parts": [{
        "name": "blade",
        "color": ["light_grey"],
        "material": ["metal"]
    },
    {
        "name": "handle",
        "material": ["plastic"]
    }]
}
"A blue scissors with a light grey metal blade and a handle made of plastic"

2:
{
    "object": "pillow",
    "color": ["blue","red"],
    "material": ["fabric"],
    "pattern": ["striped"]
}
"A striped blue and red pillow made of fabric"

3:
{
    "object": "dog",
    "parts": [{
        "name": "eye",
        "color": ["brown"]
    }
    {
        "name": "body",
        "color": ["black", "white"]
    }]
}
"A dog with black eyes and a black and white body"

4:
{
    "object": "blender",
    "parts": [
        {
            "name": "switch",
            "color": [
                "white"
            ],
            "material": [
                "plastic"
            ]
        },
        {
            "name": "base",
            "color": [
                "dark_red",
                "light_grey"
            ],
            "material": [
                "metal"
            ]
        }
    ]
}
 "A blender with a white plastic switch and a dark red and light grey metal base."
 
Use only one sentence, and avoid using subjective adjectives and information not present in the json. Be verbose and pedantic, using all the attributes in the json. 

You are ready?
"""
  
    answer = """Yes i am ready! Give me your json file."""
    obj['prompt'] = question
    obj['answer'] = answer
    # obj['new_prompt'] = json.dumps(keep_only_attributes(simplify_attributes(obj)), indent=4)
    obj['new_prompt'] = json.dumps(keep_only_attributes(simplify_attributes(obj)), indent=4)
    return obj


def simplify_attributes(object):
    """Transform the attributes from lists to only one value"""
    obj = copy.deepcopy(object) # avoiding to modify original object
    
    for key, value in object.items():
        if key in ['color', 'material', 'pattern', 'transparency']:
            if value == []:
                obj.pop(key)
            else:
                obj[key] = value
                
        if key == 'parts':
            for i, part in enumerate(object['parts']):
                obj['parts'][i] = copy.deepcopy(part)
                for key_part, value_part in object['parts'][i].items():
                    if key_part in ['color', 'material', 'pattern', 'transparency']:
                        if value_part == []:
                            del obj['parts'][i][key_part]
                        else:
                            obj['parts'][i][key_part] = value_part
    assert 'parts' not in obj or len(obj['parts']) == len(object['parts']), "Number of parts not congruent"
    
    return obj
            


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("tokenizers/open_assistant", padding_side="letft")
    model = AutoModelForCausalLM.from_pretrained("models/open_assistant")
        
    # tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/oasst-sft-1-pythia-12b")
    # model = AutoModelForCausalLM.from_pretrained("OpenAssistant/oasst-sft-1-pythia-12b")
    # tokenizer.save_pretrained("tokenizers/open_assistant", from_pt=True) 
    # model.save_pretrained("models/open_assistant", from_pt=True) 
    
    model.half().cuda()
    return model, tokenizer

def get_answer(model, tokenizer, prompter_text="", assistant_text = "\"A ", text=None):
    text =  "<|prompter|>%s<\s><|assistant|>%s" % (prompter_text, assistant_text) if text is None else text
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.cuda()
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_new_tokens=80)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text

def get_answer_batched(model, tokenizer, bs=32, prompter_text=None, assistant_text=None, text=None):
    text = ["<|prompter|>%s<\s><|assistant|>%s" % (prmpt, assist) for prmpt, assist in zip(prompter_text, assistant_text)] if text is None else text
    
    prompts = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    prompts['input_ids'] = prompts['input_ids'].cuda()
    prompts['attention_mask'] = prompts['attention_mask'].cuda()
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    with torch.no_grad():
        gen_tokens = model.generate(
        **prompts,
        do_sample=True,
        temperature=0.8,
        max_new_tokens=80,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)
    return gen_text

def get_question(object):
    obj = simplify_attributes(object.copy())
    prompt = 'Can you provide one caption of an object described by the following structured attributes: "'
    for key, value in obj.items():
        if key not in ['object', 'parts', 'color', 'material', 'transparency', 'pattern'] or value == []:
            continue
        if key != 'object':
            prompt += '; '
        if key != 'parts':
            prompt += "%s: '%s'" % (key, value)
        else:
            prompt += '"and with a part described by:"'
            for key, value in obj['parts'][0].items():
                if value == []:
                    continue
                if key != 'name':
                    prompt += '; '
                if key != 'parts':
                    prompt += "%s: '%s'" % (key, value)
                    
    return prompt + '"? Do not the describe what the object is but focus only on its attributes and visual information'

def get_question_from_caption(obj):
    return 'Use synonyms to enrich this caption: \"%s\"' % obj['positive_caption']

def add_part(prompt, part):
    prompt += "<|prompter|>The above object comprises also the following details: \""
    for key, value in part.items():
        if key != 'name':
            prompt += ';'
        if key != 'parts':
            prompt += " %s: '%s'" % (key, value)
    return prompt + '? Can you add them?'

def clean_response(response):
    response = response.replace('\n', ' ')
    response = response.replace('\t', ' ')
    response = response.replace('  ', ' ')
    response = response.replace('<\s>', '')
    response = remove_non_ascii(response)
    return response

def get_last_answer(conversation, replacing=True):
    start = '<|assistant|>'
    end = '<\s>'
    last_start = conversation.rfind(start)
    last_end = conversation.rfind(end) if conversation.rfind(end) > last_start else conversation.rfind('</s>')
    answer = conversation[last_start + len(start):last_end] if last_start < last_end else conversation[last_start + len(start):]
    if replacing:
        answer = answer.replace('"', '')
        answer = answer.replace('.', '')
    return answer

def keep_only_attributes(obj):
    attributes = ['object', 'color', 'material', 'transparency', 'pattern', 'parts']
    new_obj = {}
    for key, value in obj.items():
        if key in attributes:
            new_obj[key] = value
            
    return new_obj

def visualize_conversations(objs, show_json=True, show_question=True, new_lines=2, show_only_last_question=False):
    objs = [objs] if type(objs) != list else objs
    
    out = ''
    
    for i, obj in enumerate(objs):
        
        out += 'Conversation %d:\n' % i if show_question else ''
        out += '%s\n' % json.dumps(keep_only_attributes(obj), indent=4) if show_json else ''
        
        if not show_only_last_question:
            prompts = [obj['prompt']] if 'old_prompts' not in obj else obj['old_prompts'] + [obj['prompt']]
            answers = [obj['answer']] if 'old_answers' not in obj else obj['old_answers'] + [obj['answer']]
        else:
            prompts = [obj['prompt']]
            answers = [obj['answer']]      
        
        for prompt, answer in zip(prompts, answers):
            out += '%s\n%s\n' % (prompt, answer)
        out += '\n' * new_lines
            
    return out


def create_all_conversation(obj):
    assert not ('new_prompt' not in obj or obj['new_prompt'] == ''), "There is no old conversation"
    
    text = ""
    if 'old_prompts' in obj:
        for prompt, answer in zip(obj['old_prompts'], obj['old_answers']):
            text += '<|prompter|>%s<\s><|assistant|>%s<\s>' % (prompt, answer)
        
    text += '<|prompter|>%s<\s><|assistant|>%s<\s>' % (obj['prompt'], obj['answer'])
    text += '<|prompter|>%s<\s><|assistant|>"A ' % obj['new_prompt']
    
    return text

def get_captions_one_part_batched(model, tokenizer, objects, limit=-1, batch_size=32, is_a_continuation=False, context_question=False):
    text = ""
    
    limit = limit if limit > -1 else len(objects)
    
    index = 0
    
    if context_question:
        is_a_continuation = True
    
    assistant_text = ["\"A "] * batch_size
    saved_obj = []
    prompts = []
    conversations = []
    
    for i, obj in enumerate(tqdm((objects[:limit]))):
        # saved_obj.append(obj.copy())
        
        
        if not is_a_continuation:
            prompts += [get_question(obj)] 
        else:
            # creates the context question and answer
            if context_question:
                obj = create_context(obj)
            saved_obj.append(obj.copy())
            conversations += [create_all_conversation(obj)]
            prompts += [obj['new_prompt']]
        # check if we have to start a new batch
        if ((i + 1) % batch_size == 0) or (i + 1) == limit:
            bs = batch_size if (i + 1) != limit else limit - index
            
            if not is_a_continuation:
                texts = get_answer_batched(model, tokenizer, bs, prompts[:bs], assistant_text[:bs])
            else:
                texts = get_answer_batched(model, tokenizer, bs, text=conversations)
            for j, text in enumerate(texts):
                resp = clean_response(get_last_answer(text, False))
                
                #store old prompts and command in case it is not the first question
                if 'new_prompt' in obj:
                    saved_obj[index]['old_prompts'] = [saved_obj[index]['prompt']] if 'old_prompts' not in saved_obj[index] else saved_obj[index]['old_prompts'] + [saved_obj[index]['prompt']]
                    saved_obj[index]['old_answers'] = [saved_obj[index]['answer']] if 'old_answers' not in saved_obj[index] else saved_obj[index]['old_answers'] + [saved_obj[index]['answer']]
                    
                #saving the prompt and the answer
                saved_obj[index]['prompt'] = prompts[j]
                saved_obj[index]['answer'] = resp
                
                
                index += 1

            conversations = []
            prompts = []
        
    #saveObject(saved_obj, 'responses/captioned_objs')
    return saved_obj

def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else '' for i in text])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number to use')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to process')
    args = parser.parse_args()
    
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(SEED) # setting fixed seed to make experiments reproducible
    
    objects = loadObject("pickle/%s_prepared" % args.dataset)
    #objects = loadObject("responses/captioned_objs_short")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("tokenizers/open_assistant", padding_side="left")
    model = loadObject("models/open_assistant/model")
    model.cuda()
    print('done')
    
    saved_objs = get_captions_one_part_batched(model, tokenizer, objects, batch_size=args.batch_size)
    saveObject(saved_objs, 'pickle/%s_captioned')
    text = visualize_conversations(saved_objs)
    with open('responses/responses.txt', 'w') as f:
        f.write(remove_non_ascii(text))


if __name__ == "__main__":
    main()