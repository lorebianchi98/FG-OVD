import argparse
import json
import os
import shutil
import numpy as np

import torch
import yaml

from PIL import Image
from transformers import AutoProcessor, OwlViTForObjectDetection, Owlv2ForObjectDetection

from copy import deepcopy
from eval.evaluate_map import read_json
from src.losses import PushPullLoss
from src.dataset import remove_unprocessable_entries, get_dataloaders, keep_only_rare
from src.models import PostProcess, load_model
from src.train_util import train, validate, validate_lvis, validate_lvis_hf, get_ids_per_frequencies
from src.util import BoxUtil, GeneralLossAccumulator, ProgressFormatter, ModelUtil, get_processor, process_single_string
from torch.utils.tensorboard import SummaryWriter

def get_training_config(config_path):
    with open(config_path, "r") as stream:
        data = yaml.safe_load(stream)
        data['training']['n_accumulation_steps'] = data['training'].get('n_accumulation_steps', 1)
        data['training']['self_distillation'] = data['training'].get('self_distillation', False)
        return data["training"]
    
def get_data_config(config_path):
    with open(config_path, "r") as stream:
        data = yaml.safe_load(stream)
        return data["data"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/base_config.yaml", help="Training configuration file to load")
    parser.add_argument('--do_not_use_hf_lvis_evaluation', action='store_true', help="If setted, the evaluation on LVIS will be performed without the HuggingFace interface and by using the model as it is trained.") 
    parser.add_argument('--out', type=str, default="result", help="Base OWL model to use") 
    args = parser.parse_args()
    
    # setting random seed in order to make reproducible results
    SEED = 123
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    general_loss = GeneralLossAccumulator()
    progress_summary = ProgressFormatter()

    postprocess = PostProcess(confidence_threshold=0, iou_threshold=0.5)

    model_name = args.out.split('/')[-1]
    writer = SummaryWriter(f'runs/{model_name}_')
    
    training_cfg = get_training_config(args.config)
    data_cfg = get_data_config(args.config)
    lvis_evaluation = 'lvis_annotations_file' in data_cfg
    train_dataloader, val_dataloader, lvis_dataloader = get_dataloaders(data_cfg, training_cfg)
    model = load_model(device, training_cfg['base_model'])
    processor = get_processor(training_cfg['base_model'])

    # We load all the ground truth for the validation set in the format to evaluate_map
    print("Loading Validation dataset")
    val_data = read_json(data_cfg['test_annotations_file'])
    val_data = remove_unprocessable_entries(val_data, training_cfg, perform_cleaning=True)
    
    queries = None
    if lvis_evaluation or training_cfg['self_distillation']:
        lvis_data = read_json("lvis/lvis_v1_val.json") # read_json(data_cfg['lvis_annotations_file'])
        # validation of the model without hugging face interface
        if args.do_not_use_hf_lvis_evaluation or training_cfg['self_distillation']:

            with torch.no_grad():
                vocabulary = ['a ' + process_single_string(cat['name']) for cat in lvis_data['categories']]
                inputs = processor(
                    text=[vocabulary],
                    images=Image.new("RGB", (224, 224)),
                    return_tensors="pt",
                    padding=True
                )
                inputs.to(device)
                queries = model.text_embedder(inputs)
                queries = queries.unsqueeze(0)
            ids_per_frequencies = get_ids_per_frequencies(lvis_data)
        # textual queries for hugging face inference
        text_query = ['a ' + cat['name'] for cat in lvis_data['categories']]
        
        # we create a temporary dir
        tmp_base_path = os.path.join('tmp', args.out.split('/')[-1])
        os.makedirs(tmp_base_path, exist_ok=True)

        

    criterion = PushPullLoss(
        training_cfg['n_hardnegatives'] + 1,
        margin=training_cfg['margin'],
        self_distillation_loss=training_cfg.get('self_distillation', 'mse'),
        class_ltype=training_cfg.get('ltype', 'triplet')
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=training_cfg["weight_decay"],
    )
    
    best_map = 0
    for epoch in range(-1, training_cfg["n_epochs"]):
        # train loop
        model.train()
        train_metrics = train(model,
                              train_dataloader,
                              criterion,
                              optimizer,
                              general_loss,
                              epoch,
                              training_cfg['n_accumulation_steps'],
                              lvis_query_embeds=queries,
                              writer=writer) if epoch > -1 \
                                      else {'loss_triplet': 0, 'loss_bg': 0, 'loss_bbox': 0, 'loss_giou': 0}
        
        # eval loop
        model.eval()
        val_metrics = validate(model, val_dataloader, deepcopy(val_data), epoch, writer)
        # print training summary
        progress_summary.update(epoch, train_metrics, val_metrics)
        progress_summary.print()
        if lvis_evaluation:
            print("Evaluating LVIS...")
            if args.do_not_use_hf_lvis_evaluation:
                # WARN: this case is not fully tested and it may contain errors
                lvis_metrics = validate_lvis(model, lvis_dataloader, queries, postprocess, ids_per_frequencies, epoch, writer)
            else:
                # we want to evaluate the model on LVIS using the huggingface interface
                # to do so, we save a temporary file of the weights of the model and we load them inside the hf model
                tmp_model_path = os.path.join(tmp_base_path, 'model_tmp')
                ModelUtil.create_base_model(model, training_cfg['base_model']).save_pretrained(tmp_model_path)
                # loading huggingface model
                if 'owlv2' in training_cfg['base_model']:
                    hf_model = Owlv2ForObjectDetection.from_pretrained(tmp_model_path)
                elif 'owlvit' in training_cfg['base_model']:
                    hf_model = OwlViTForObjectDetection.from_pretrained(tmp_model_path)
                hf_model.to(device)
                
                lvis_metrics = validate_lvis_hf(hf_model, processor, data_cfg, training_cfg, lvis_data, text_query, tmp_base_path, epoch, writer)
            
            print("LVIS " + str(lvis_metrics['map']))
        
        if val_metrics['map'] > best_map and epoch > -1:
            print("Best validation mAP, saving the weights...")
            best_map = val_metrics['map']
            ModelUtil.create_base_model(model, training_cfg['base_model']).save_pretrained(args.out)
        print(f"Saving model at epoch {epoch}...")
        out_path = args.out + f'_epoch{epoch}'
        ModelUtil.create_base_model(model, training_cfg['base_model']).save_pretrained(out_path)

if __name__ == "__main__":
    main()