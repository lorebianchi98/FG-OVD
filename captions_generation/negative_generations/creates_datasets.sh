#!/bin/bash

# Input and output directories + n_hard_negatives
input_dir=$1
output_dir=$2
n_hardnegatives=$3

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# shuffle negatives
output_file="$output_dir/shuffle_negatives.json"
echo "Running: python3 main.py --input_dir \"$input_dir\" --output_file \"$output_file\" --n_hardnegatives \"$n_hardnegatives\" --shuffle_negatives"
python3 main.py --input_dir "$input_dir" --output_file "$output_file" --n_hardnegatives "$n_hardnegatives" --shuffle_negatives

# Loop through attribute changes
for num_attributes in {1..3}; do
    output_file="$output_dir/${num_attributes}_attributes.json"
    echo "Running: python3 main.py --input_dir \"$input_dir\" --output_file \"$output_file\" --n_hardnegatives \"$n_hardnegatives\" --n_attributes_change $num_attributes"
    python3 main.py --input_dir "$input_dir" --output_file "$output_file" --n_hardnegatives "$n_hardnegatives" --n_attributes_change $num_attributes 
done

# Attributes to change
attributes=("color" "material" "pattern" "transparency")
# Loop through attribute mask
for attribute in "${attributes[@]}"; do
    output_file="$output_dir/${attribute}.json"
    echo "Running: python3 main.py --input_dir \"$input_dir\" --output_file \"$output_file\" --n_hardnegatives \"$n_hardnegatives\"  --to_change \"$attribute\""
    python3 main.py --input_dir "$input_dir" --output_file "$output_file" --n_hardnegatives "$n_hardnegatives" --to_change "$attribute"
done
