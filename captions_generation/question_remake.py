from open_assistant import get_captions_one_part_batched, saveObject, loadObject, visualize_conversations, keep_only_attributes, remove_non_ascii
import argparse
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 123

import os


def remove_ok_captions(objs):
    new_objs = []
    ok = []
    for obj in objs:
        if obj['new_prompt_id'] != -1:
            new_objs += [obj]
        else:
            ok += [obj]  
    return new_objs, ok

def reask_caption(model, tokenizer, objs, batch_size=32):
    return get_captions_one_part_batched(model, tokenizer, remove_ok_captions(objs)[0], batch_size=batch_size, is_a_continuation=True)
    
def main():
    
    
    
    objs = reask_caption(objs,batch_size=16)
    saveObject(objs, 'objs_reasked')
    # objs = loadObject('objs_reasked')
    text = visualize_conversations(objs)
    with open('responses/2-step_responses.txt', 'w') as f:
        f.write(remove_non_ascii(text))
        
if __name__ == '__main__':
    main()