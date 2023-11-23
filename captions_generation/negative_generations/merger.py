import json
import os
import argparse

import re

def merge_dataset(input_dir, output_file):
    images = []
    annotations = []
    categories = []
    added_category_ids = set()
    added_image_ids = []
    # Get a list of split files in the input directory
    file_list = sorted(os.listdir(input_dir))

    # Sort the file list based on the numerical part of the filename
    # file_list = sorted(file_list, key=lambda x: int(re.findall(r'\d+', x)[0]))

    # Iterate over the sorted file list
    for filename in file_list:
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)

            # Load JSON data from each file
            with open(file_path, "r") as file:
                data = json.load(file)

            # Merge images and annotations  
            for image in data['images']:
                image_id = image['id']
                if image_id not in added_image_ids:
                    added_image_ids.append(image_id)
                    images.append(image)
            
            annotations.extend(data["annotations"])

            # Merge categories, checking for duplicates
            for category in data["categories"]:
                category_id = category["id"]
                if category_id not in added_category_ids:
                    categories.append(category)
                    added_category_ids.add(category_id)

    # Create merged dataset
    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save merged dataset to output file
    with open(output_file, "w") as file:
        json.dump(dataset, file, indent=0)


    print(f"Merged dataset saved to {output_file}.")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge split JSON object detection dataset into a single file.")
    parser.add_argument("--input_dir", type=str, help="Input directory containing the split dataset files.")
    parser.add_argument("--output_file", type=str, help="Output file to save the merged dataset.")
    args = parser.parse_args()

    merge_dataset(args.input_dir, args.output_file)
