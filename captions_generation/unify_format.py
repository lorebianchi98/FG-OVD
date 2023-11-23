from utils.json_handler import read_json, write_json
from tqdm import tqdm
from prepare_for_assistant import check_equality
from utils.data_handler import group_image_per_filepath
def transform_predslist_to_dict_by_image(preds):
    """Group predictions by image path"""
    result = {}
    for pred in preds:
        image = pred['image']
        if image not in result:
            result[image] = []
        result[image].append(pred)
    return result  

def retrieve_image_from_filename(filename: str, images: list) -> dict:
    for image in images:
        if image['file_name'] == filename:
            return image
    return None

def get_caption_id(new_caption, categories):
    for category in categories:
        if new_caption == category['name']:
            return category['id']
    else:
        return -1
    
def update_categories(cat_ids, categories):
    for category in categories:
        if category['id'] in cat_ids:
            category['instance_count'] += cat_ids.count(category['id'])
            category['image_count'] += 1
    
def group_objects_by_image(objects):
    result = {}
    for obj in objects:
        image = obj['image']
        if image not in result:
            result[image] = []
        result[image].append(obj)
    return result

def find_filling(src, targets):
    """Search if the object src is equal (same attributes and parts) to some object in targets, and it is also in the same image"""
    
    for target in targets:
        if check_equality(src, target) == 0:
            return target
    
    return {}

def fill_objects(targets, srcs):
    """
    Inserts objects in the list src to the main object list, by copying positives and negative captions from the corresponding objects.
    To call before unify_format.
    """
    
    targets_per_image = group_objects_by_image(targets)
    for src in tqdm(srcs):
        if src['image'] in targets_per_image:
            target = find_filling(src, targets_per_image[src['image']])
            if target == {}:
                continue
            src['positive_caption'] = target['positive_caption']
            src['negative_captions'] = target['negative_captions']
            targets.append(src)

    return targets

def prepare_json_to_revision(data):
    images = []
    annotations = []
    for imm in data['images']:
        imm['checked'] = True # setted to True only to let the generations of negative possible without revision (for blind reviews)
        images.append(imm)
    
    for ann in data['annotations']:
        ann['needs_revision'] = False
        annotations.append(ann)
        
    data['images'] = images
    data['annotations'] = annotations
    
    return data
        

def unify_format(objects: list, original_json: dict, n_hardnegatives=10, one_sample_per_image=False, limit=None):
    """
    Creates a dict in the standard lvis format
    {  
        info: info
        images: [images],
        annotations: [annotations],
        licenses: [licenses],
    }  

    info{  
        year: int
        version: str,
        description: str,
        contributor: str,
        url: str,
        date_created: datetime,
    }  

    license{  
        id: int
        name: str,
        url: str,
    }  
    
    image{  
        id: int
        width: int,
        height: int,
        license: int,
        flickr_url: str,
        coco_url: str,
        date_captured: datetime,
        not_exhaustive_category_ids: [int],
        neg_category_ids: [int],
    }  
    
      categories{  
        id: int
        synset: str,
        synonyms: [str], # we could insert captions described by the same object attributes
        def: str,
        instance_count: int,
        image_count: int,
        frequency: str,
    }  
    
    """
    data = group_objects_by_image(objects)
    images = []
    annotations = []
    categories = []
    last_category_id = -1
    last_annotation_id = -1
    # iterate over each image
    count = 0
    images_dict = group_image_per_filepath(original_json['images'])
    for filename, objs in tqdm(data.items()):
        if limit is not None and count > limit:
            break
        img = images_dict[filename]
        assert img != None, "Found unexisting image"
        
        img_categories = []
        img_neg_categories = []
        
        categories_set = {} # set of categories ids for this image, the key is the positive caption
        for obj in objs:
            if len(obj['negative_captions']) == 0:
                continue
            ann_neg_categories = []
            # captions are inserted inside categories if they do not already exist
            capts = [obj['positive_caption']] + obj['negative_captions'][:n_hardnegatives]
            # if object is not filled we create categories object
            if not obj['positive_caption'] in categories_set:
                for i, capt in enumerate(capts):
                    last_category_id += 1
                    capt_id = last_category_id
                    # creating category object
                    category = {'name': capt, 'id': last_category_id, 'synset': '', 'synonyms': [], 'def': '', 'frequency': 'f', 'instance_count': 0, 'image_count': 0}
                    categories.append(category)
                    categories_set[obj['positive_caption']] = [capt_id] if obj['positive_caption'] not in categories_set else categories_set[obj['positive_caption']] + [capt_id]
                    if i == 0:
                        annotation_category_id = capt_id #setting the category id to set inside the annotation
                        img_categories += [capt_id]
                    else:
                        img_neg_categories += [capt_id] if capt_id not in img_categories else []
                        ann_neg_categories += [capt_id]
            else:
                annotation_category_id = categories_set[obj['positive_caption']][0]
                ann_neg_categories = categories_set[obj['positive_caption']][1:]

            #creating the annotation object
            last_annotation_id += 1
            assert len(ann_neg_categories) <= n_hardnegatives, "too much hard negatives to add"
            assert len(ann_neg_categories) > 0 or n_hardnegatives == 0, "zero hard negatives to add"
            
            if limit is None:
                annotation = {'bbox': obj['bbox'], 'area': obj['area'], 'segmentation': obj['segmentation'], 'category_id': annotation_category_id, 'neg_category_ids': ann_neg_categories, 'image_id': img['id'], 'needs_revision': False, 'id': last_annotation_id}
            else:
                # if a limit is setted then we add the negative captions in order to visualize it inside FiftyOne
                annotation = {'bbox': obj['bbox'], 'area': obj['area'], 'segmentation': obj['segmentation'], 'category_id': annotation_category_id, 'neg_category_ids': ann_neg_categories, 'neg_captions': ann_neg_captions, 'image_id': img['id'], 'needs_revision': False, 'id': last_annotation_id}

            if one_sample_per_image:
                # adding one image for each annotation
                update_categories(img_categories, categories)
                img_categories  = list(set(img_categories))
                img_neg_categories = list(set(img_neg_categories))
                
                
                #creating image object
                image = {'id': count, 'width': img['width'], 'height': img['height'], 'license': img['license'], 'coco_url': img['coco_url'], 'flickr_url': img['flickr_url'], 'file_name': filename, 'neg_category_ids': img_neg_categories, 'not_exhaustive_category_ids': img_categories}  
                images.append(image) 
                
                annotation['image_id'] = count
                
                img_categories  = []
                img_neg_categories = []
                count += 1
                
                if limit is not None and count > limit:
                    annotations.append(annotation)
                    break
            
            annotations.append(annotation)
        
        # updating image information
        if not one_sample_per_image:
            update_categories(img_categories, categories)
            img_categories  = list(set(img_categories))
            img_neg_categories = list(set(img_neg_categories))
            
            #creating image object
            image = {'id': img['id'], 'width': img['width'], 'height': img['height'], 'license': img['license'], 'coco_url': img['coco_url'], 'flickr_url': img['flickr_url'], 'file_name': filename, 'neg_category_ids': img_neg_categories, 'not_exhaustive_category_ids': img_categories, 'checked': False}  
            images.append(image) 
            
        count += 1 if not one_sample_per_image else 0
            
    #return {'info': original_json['info'], 'categories': categories, 'annotations': annotations, 'images': images, 'licenses': original_json['licenses']}
    return {'categories': categories, 'annotations': annotations, 'images': images}

def main():
    new_path = 'json/paco_lvis_v1_test_captioned.json'
    data = read_json('json/paco_lvis_v1_test_captioned_objs.json')
    original_json = read_json('json/paco_lvis_v1_test.json')
    new_data = unify_format(data, original_json)
    write_json(new_data, new_path)
    
if __name__ == "__main__":
    main()