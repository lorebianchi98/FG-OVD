# PacoDatasetHandling
Scripts used to manipulate the PACO dataset and interact with OpenAssistant to create a dataset of captions (correlated with hard negatives) for LVIS detections.
## Installation
To perform the dataset collection it will be necessary to create a Docker container using the following commands:
```bash
git clone https://github.com/lorebianchi98/PacoDatasetHandling.git
cd PacoDatasetHandling/docker/
docker image build -t lorenzobianchi/paco_dataset_handling - < Dockerfile
```
# Creates dataset
Creates the whole dataset by manipulating PACO json and interacting with OpenAssistant.
```bash
python main.py --gpu DEVICE_ID --dataset paco_lvis_v1_test --batch_size BATCH_SIZE 
```
We used a batch size of 4. This command will create a benchmark with 10 temporary hardnegatives of Hard type.
To create the hardnegatives of Hard, Medium, Easy, Trivial, Color, Material and Transparency type, with 10 hardnegatives, it is necessary to run the following commands:
```bash
cd negative_generations
./creates_datasets.sh ../jsons ../OUT_DIR 10
```

# Dataset format
The dataset follows the standard LVIS format:
```python
data["images"]: # a list of dictionaries, each dictionary corresponds to one image
{
    'id':                                   int,
    'file_name':                            str,
    'width':                                int,
    'height':                               int,
    'neg_category_ids':                     List[int], 
    'not_exhaustive_category_ids':          List[int], # not used
    'neg_category_ids_attrs':               List[int], # not used
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
