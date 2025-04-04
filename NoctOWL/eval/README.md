# Evaluation
To have the results of NoctOWL that are reported on our paper we need to calculate the ranks and mAPs for every benchmark for every possible number of negatives captions and evaluate the model on LVIS validation set.

## FG-OVD evaluation
We have to generate the predictions files, and evaluate them. Consider to have your trained model weights at path/to/model/weights/model_name, run the following commands
```bash
# generate predictions in output_base in a folder equal to the name of the weights folder
python results.py --model path/to/model/weights/model_name --output_base predictions_fgovd --tokenizer google/owlvit-base-patch16 # choose the correct tokenizer (depending of the base model) from google/owlvit-base-patch16, google/owlv2-base-patch16, google/owlvit-large-patch14, google/owlv2-large-patch14 

# eval rank
python run_all_ranks.py --model_predictions predictions_fgovd/model_name --output_base ranks

# eval mAPS
python run_all_predictions.py --model_predictions predictions_fgovd/base-model_name --output_base maps
```

## LVIS Evaluation
We refer to the (original OWL-ViT repo)[https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit] to evaluate NoctOWL weights on LVIS validation set. Follow the installation instruction of OWL-ViT in the original repository to create an appropriate conda environment, and place the script of this repo `eval/lvis/evaluator.py` in the place of the `evaluator.py` of the original repository. The script in this repo copy the weights of the trained layer of NoctOWL (depending of the argument --noctowl_weigths) to the OWL tensorflow interface used in the original repository and run the original evaluation code.
An example of execution of the evaluation of NoctOWLv2 Base is the following:
```bash
python -m scenic.projects.owl_vit.evaluator \
  --alsologtostderr=true \
  --platform=gpu \
  --config=owl_v2_clip_b16 \
  --checkpoint_path=weights/owl2-b16-960-st-ngrams_c7e1b9a \
  --annotations_path=annotations/lvis_v1_val.json \
  --tfds_data_dir=data_dir \
  --output_dir=tmp \ 
  --noctowl_weights=lorebianchi98/NoctOWLv2-base-patch16
```