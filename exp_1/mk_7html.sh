#!/bin/bash

# Define the folders to process
folders=(
    "myFlickr30k_ep100"
    "myFlickr30k_mobile_ep100"
    "myFlickr30k_resnet101"
    "myFlickr30k_resnet152"
    "myFlickr30k_resnet50"
    "myFlickr30k_resnext101_ep100"
    "myFlickr30k_vgg_ep100"
)

# Base directory paths (modify if needed)
image_dir="./sample_images/"
script_path="./mk_html.py"

# Process each folder
for folder in "${folders[@]}"; do
    # Input and output paths
    text_file="./${folder}/predictions.txt"
    output_file="./${folder}/predictions.html"
    
    echo "Processing folder: $folder"
    echo "Input file: $text_file"
    echo "Output file: $output_file"
    
    # Run the command with time measurement
    time python3.10 "$script_path" \
        --text_file "$text_file" \
        --image_dir "$image_dir" \
        --output_file "$output_file"
    
    echo "----------------------------------------"
done

echo "All folders processed!"

