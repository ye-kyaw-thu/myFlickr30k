"""
for extract image filenames from predictions.txt and copying
written by Ye, LST Lab., Myanmar
last updated: 3 July 2025
"""

import os
import shutil
import argparse
import re

def extract_image_filenames(text_file):
    """Extract image filenames from predictions.txt file"""
    image_filenames = []
    with open(text_file, 'r', encoding='utf-8') as f:
        content = f.read()
        # Use regex to find all lines like "Image: 3139876823.jpg"
        matches = re.findall(r'Image:\s+([^\s]+\.jpg)', content)
        image_filenames = list(set(matches))  # Remove duplicates if any
    return image_filenames

def copy_images(image_list, src_dir, dest_dir):
    """Copy images from source to destination directory"""
    copied_count = 0
    os.makedirs(dest_dir, exist_ok=True)
    
    for filename in image_list:
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"Copied: {filename}")
            copied_count += 1
        else:
            print(f"Warning: Source file not found - {filename}")
    
    return copied_count

def main():
    parser = argparse.ArgumentParser(
        description='Copy images listed in predictions file to a new directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--text_file', required=True, 
                       help='Path to predictions.txt file containing image names')
    parser.add_argument('--src_dir', required=True, 
                       help='Source directory containing original images')
    parser.add_argument('--dest_dir', required=True, 
                       help='Destination directory to copy images to')
    
    args = parser.parse_args()
    
    print(f"\nExtracting image filenames from: {args.text_file}")
    image_filenames = extract_image_filenames(args.text_file)
    print(f"Found {len(image_filenames)} unique image references in the file")
    
    print(f"\nCopying images from: {args.src_dir}")
    print(f"Copying images to: {args.dest_dir}\n")
    
    copied_count = copy_images(image_filenames, args.src_dir, args.dest_dir)
    
    print(f"\nDone! Copied {copied_count} images out of {len(image_filenames)}")
    if copied_count < len(image_filenames):
        print(f"Warning: {len(image_filenames) - copied_count} images were not found in source directory")

if __name__ == "__main__":
    main()

