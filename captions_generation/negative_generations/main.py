from creates_hardnegatives import create_hardnegatives_corrected_captions, loadObject, write_json, fill_missing, delete_missing
from merger import merge_dataset

import argparse
import random
from tqdm import tqdm

SEED = 123


def clean_data(data):
    """
    Remove images that have not been checked.
    Remove deleted annotations.
    Remove images that have no annotations.
    Remove categories with no associated annotations (both as positive or as negative)
    Remove annotations with no images.
    """
    
    # filtering out unchecked images
    new_images = [imm for imm in data['images'] if imm['checked']]
    data['images'] = new_images
    
    # filtering out deleted annotations
    new_annotations = [ann for ann in data['annotations'] if ann['needs_revision'] == False]
    data['annotations'] = new_annotations
    
    # filtering out images without annotations
    id_annotations = [ann['image_id'] for ann in data['annotations']]
    new_images = [imm for imm in data['images'] if imm['id'] in id_annotations]
    data['images'] = new_images
    
    # filtering out annotations for deleted images
    imm_ids = [imm['id'] for imm in data['images']]
    new_annotations = [ann for ann in data['annotations'] if ann['image_id'] in imm_ids]
    data['annotations'] = new_annotations
    
    # filtering out annotations with no hnegatives
    new_annotations = [ann for ann in data['annotations'] if len(ann['neg_category_ids']) > 0]
    print("Deleted %d annotations out of %d" % (len(data['annotations']) - len(new_annotations), len(data['annotations'])))
    data['annotations'] = new_annotations
    
    # filtering out images without annotations
    id_annotations = [ann['image_id'] for ann in data['annotations']]
    new_images = [imm for imm in data['images'] if imm['id'] in id_annotations]
    data['images'] = new_images
    
    # filtering out categories without associations
    id_categories_associated = []
    for ann in data['annotations']:
        id_categories_associated += [ann['category_id']] + ann['neg_category_ids']
    new_categories = [cat for cat in data['categories'] if cat['id'] in id_categories_associated]
    data['categories'] = new_categories
    return data
    
    
def creates_random_negatives_from_dataset(data):
    """
    
    """
    OBJECTS = ['trash_can','handbag','ball','basket','bicycle','book','bottle','bowl','can','car_(automobile)','carton','cellular_telephone','chair','cup','dog','drill','drum_(musical_instrument)', 'guitar','hat','helmet','jar','knife','laptop_computer','mug','pan_(for_cooking)','plate','remote_control','scissors','shoe','slipper_(footwear)','stool','table','towel','wallet','watch','wrench','belt','bench','blender','box','broom','bucket','calculator','clock','crate','earphone','fan','hammer','kettle','ladder','lamp','microwave_oven','mirror','mouse_(computer_equipment)','napkin','newspaper','pen','pencil','pillow','pipe','pliers','scarf','screwdriver','soap','sponge','spoon','sweater','telephone','television_set','tissue_paper','tray','vase']
    OBJECTS = [elem.split('_')[0] for elem in OBJECTS]
    
    done_categories = []
    old_categories = data['categories'].copy()
    data['categories'] = fill_missing(data['categories'])
    
    captions = []
    new_objs = []
    # keeping only the objects which are present in the dataset of captions
    for obj in OBJECTS:
        to_append = [cat['name'] for cat in old_categories if ' ' + obj + ' ' in cat['name'] and len(cat['name'].split(' ')) < 13]
        if to_append == []:
            continue
        captions.append(to_append)
        new_objs.append(obj)
    
    print("Shuflling negatives...")
    for ann in tqdm(data['annotations']):
        if ann['category_id'] in done_categories:
            continue
        done_categories += [ann['category_id']]
        
        to_search_objects = random.sample(range(len(captions)), len(ann['neg_category_ids']))
        for i in range(len(ann['neg_category_ids'])):
            # we check if the object of the new caption is different from the one of the positive caption
            if ' ' + new_objs[to_search_objects[i]] + ' ' in data['categories'][ann['category_id']]['name']:
                to_search_objects[i] = random.choice([x for x in range(len(new_objs)) if x not in to_search_objects])
            categories = captions[to_search_objects[i]]
            
            new_caption = random.choice(categories)
            data['categories'][ann['neg_category_ids'][i]]['name'] = new_caption
            
    data['categories'] = delete_missing(data['categories'])
    return data
            


def main():
    parser = argparse.ArgumentParser(description="Merge split JSON object detection dataset into a single file, creates the hardnegatives for all the categories and eliminates unchecked images")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing the split dataset files.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save the merged dataset.")
    parser.add_argument("--n_hardnegatives", type=int, default=10, help="Number of hardnegatives to create for each category.")
    parser.add_argument("--n_attributes_change", type=int, default=1, help="Number of attributes to change for each category.")
    parser.add_argument("--shuffle_negatives", action='store_true', default=False, help="If setted, the negatives of the captions are taken randomly from the dataset")
    parser.add_argument("--to_change", nargs='+', choices=['color', 'material', 'pattern', 'transparency'], default=['color', 'material', 'pattern', 'transparency'], help="Attributes to change for hardnegatives.")
    args = parser.parse_args()

    HN_MASK = [attr in args.to_change for attr in ['color', 'material', 'pattern', 'transparency']]
    
    
    random.seed(SEED)
    # load PACO objects
    paco_objects = loadObject('../datasets/not_captioned')
    
    # merge dataset
    data = merge_dataset(args.input_dir, args.output_file)
    
    # create hardnegatives
    data = create_hardnegatives_corrected_captions(data, paco_objects, args.n_hardnegatives, hn_mask=HN_MASK, n_attributes_change=args.n_attributes_change)
    # write_json(data, '%s_uncleaned.json' % args.output_file[:-5])
    if args.shuffle_negatives:
        data = creates_random_negatives_from_dataset(data)
    # clean deleted elements
    data = clean_data(data)
    write_json(data, args.output_file)
if __name__ == '__main__':
    main()
