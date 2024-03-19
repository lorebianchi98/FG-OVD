# Fine-Grained Open Vocabulary object Detection (FG-OVD)
<span style="font-size: xx-large;">[Project page](https://lorebianchi98.github.io/FG-OVD/) | [![arXiv](https://img.shields.io/badge/arXiv-2311.17518-b31b1b.svg)](https://arxiv.org/abs/2311.17518) 

Official repository of the paper **"The devil is in the fine-grained details: Evaluating open-vocabulary object detectors for fine-grained understanding"**.

## Updates

- :fire: 02/2024: **"The devil is in the fine-grained details: Evaluating open-vocabulary object detectors for fine-grained understanding"** has been accepted to CVPR2024!


## Installation
To perform the dataset collection it will be necessary to create a Docker container using the following commands:
```bash
git clone https://github.com/lorebianchi98/FG-OVD.git
cd FG-OVD/docker
docker image build -t IMAGE_NAME - < Dockerfile
```

To use [OpenAssistant](https://github.com/LAION-AI/Open-Assistant) LLaMa-based (the model used in our experiment in the official experiments is OpenAssistant LLaMa 30B SFT 6) it is necessary to have access to LLaMa 30B by Meta AI and to obtain OpenAssistant weight following the guidelines provided [here](https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor). The model should be placed in *captions_generations/models*.

Retrieve the [PACO](https://github.com/facebookresearch/paco/tree/main) dataset and place the desired JSON file for processing into the *captions_generations/datasets* directory. In our case, we utilized paco_lvis_v1_test.json.

## Creates dataset
To create the whole dataset by manipulating PACO json and interacting with OpenAssistant.
```bash
python main.py --gpu DEVICE_ID --dataset paco_lvis_v1_test --batch_size BATCH_SIZE 
```
We used a batch size of 4. This command will create a benchmark with 10 temporary hardnegatives of Hard type.
To create the hardnegatives of Hard, Medium, Easy, Trivial, Color, Material and Transparency type, with 10 hardnegatives, it is necessary to run the following commands:
```bash
cd negative_generations
./creates_datasets.sh ../jsons ../OUT_DIR 10
```

## Dataset format
The dataset follows the standard LVIS format:
```python
data["images"]: # a list of dictionaries, each dictionary corresponds to one image
{
    'id':                                   int,
    'file_name':                            str,
    'width':                                int,
    'height':                               int,
    'license':                              int,
    'coco_url':                             str,
    'flickr_url':                           str
}

data['annotations']: # a list of dictionaries, each dictionary correspond to one annotation
{
    'id':                       int,
    'bbox':                     [x,y,width,height],
    'area':                     float,
    'category_id':              int,
    'image_id':                 int,
    'segmentation':             RLE,
    'neg_category_ids':         int, # not on LVIS
}

data["categories"]: # a list of dictionaries, each dictionary corresponds to one object category
{
    'id':               int,
    'name':             str,
    'def':              str, # always ''
    'image_count':      int,
    'instance_count':   int,
    'synset':           str, # always ''
    'synonyms':         List(str), # always []
    'frequency':        char, # always 'f'
}
```
## Reference
If you found this code useful, please cite the following paper:

      @misc{bianchi2023devil,
            title={The devil is in the fine-grained details: Evaluating open-vocabulary object detectors for fine-grained understanding}, 
            author={Lorenzo Bianchi and Fabio Carrara and Nicola Messina and Claudio Gennaro and Fabrizio Falchi},
            year={2023},
            eprint={2311.17518},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }
