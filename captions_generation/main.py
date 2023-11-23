import organizer, merger, unify_format, prepare_for_assistant, open_assistant, analyze_answers, fill_categories

from utils.pickle_handler import saveObject, loadObject
from utils.json_handler import write_json, read_json
from utils.data_handler import count_obj_with_parts, keeps_multi_attributes, show_obj, count_obj_with_parts_no_attr
import os, argparse, asyncio, sys

SEED = 123
os.environ['TRANSFORMERS_CACHE'] = 'cache/'

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', nargs='+', type=int, default=[0, 1, 2, 3], help='List of GPU numbers to use')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to process')
    parser.add_argument('--iterations', type=int, default=1, help='Number to iteration of questions to ask to OpenAssistant')
    parser.add_argument('--n_hardnegatives', type=int, default=10, help='Number of hardnegatives to generate')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    # Set the list of GPU devices to use
    device_ids = args.num_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
    
    # setting fixed seed to make experiments reproducible
    torch.manual_seed(SEED) 
    random.seed(SEED)
    
    dataset_file_name = "datasets/%s.json" % dataset_name
    # loading Open Assistant model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained("models/oasst-sft-6-llama-30b-xor/oasst-sft-6-llama-30b", padding_side="left")
    model = LlamaForCausalLM.from_pretrained("models/oasst-sft-6-llama-30b-xor/oasst-sft-6-llama-30b", device_map='auto', load_in_8bit=True)
    print('done')
    
    # creating the list of structured object in order to construct the questions for Open Assistant
    data = old_data = read_json(dataset_file_name)
    data = organizer.create_json_object_description(data)
    saveObject(data, "cache/organized")
    data = loadObject("cache/organized")
    data = merger.merge_parts(data)
    data, to_fill = prepare_for_assistant.refine_json(data)
    assert count_obj_with_parts_no_attr(data) == 0, "There are parts with no attributes unfiltered!"
    saveObject(data, "cache/prepared_%s" % dataset_name)
    saveObject(to_fill, "cache/prepared_tofill_%s" % dataset_name)
    # # asking Open Assistant the captions
    data = open_assistant.get_captions_one_part_batched(model, tokenizer, data, batch_size=args.batch_size, context_question=True)
    saveObject(data, 'pickle/%s_captioned' % dataset_name)
    data = loadObject('pickle/%s_captioned' % dataset_name)
    text = open_assistant.visualize_conversations(data, show_question=False, show_only_last_question=True, show_json=False)
    with open('responses/%s_responses.txt' % dataset_name, 'w') as f:
        f.write(open_assistant.remove_non_ascii(text))
    # asking Open Assistant to correct the captions
    data = analyze_answers.correct_captions(data, model, tokenizer, dataset_name, iterations=args.iterations, batch_size=args.batch_size)
    
    saveObject(data, 'pickle/%s_captioned_correct' % dataset_name)
    # creating hardnegatives
    data = fill_categories.create_hardnegatives(data, args.n_hardnegatives)
    saveObject(data, 'pickle/%s_captioned_correct_hn_no_fill' % dataset_name)
    text = open_assistant.visualize_conversations(data, show_json=False)
    with open('responses/%s_responses_corrected.txt' % dataset_name, 'w') as f:
        f.write(open_assistant.remove_non_ascii(text))
    
    # filling objects that needs to be checked manually
    data = unify_format.fill_objects(data, to_fill)
    saveObject(data, 'pickle/%s_captioned_correct_hn' % dataset_name)
    data = loadObject('pickle/%s_captioned_correct_hn' % dataset_name)
    data = unify_format.unify_format(data, old_data, args.n_hardnegatives)
    data = unify_format.prepare_json_to_revision(data)
    write_json(data, 'jsons/captioned_%s.json' % dataset_name)
    sys.exit(0)
if __name__ == "__main__":
    main()
