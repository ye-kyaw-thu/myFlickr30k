#!/bin/bash

# For Myanmar Flickr30k (Testing)
#time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt --output_dir ./my_ep50 --epochs 50 \
#--gpu | tee my_train_ep50.log

# For English Flickr30K (Test Run)
#time python3.10 captioner.py --data_dir ./data/flickr30k_images --caption_file captions.txt --output_dir ./ep100 --epochs 100 \
#--gpu | tee train_100.log

# Start Experiment
# For Myanmar Flickr30k (Complete version) with default resnext50
time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt --output_dir ./resnext50_ep100 --epochs 100 \
--gpu --encoder resnext50 | tee train_100_myFlickr30k_resnext50.log

        # Map user-friendly names to actual model names
#        model_map = {
#            "resnet50": "resnet50",
#            "resnet101": "resnet101",
#            "resnet152": "resnet152",
#            "mobilenetv2": "mobilenet_v2",
#            "vgg16": "vgg16",
#            "resnext50": "resnext50_32x4d",
#            "resnext101": "resnext101_32x8d"
#        }

# For Myanmar Flickr30k (Complete version) with VGG16
#time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt \
#--output_dir ./myFlickr30k_vgg_ep100 --epochs 100 --gpu --encoder vgg16 | tee train_100_myFlickr30k_vgg16.log

# For Myanmar Flickr30k (Complete version) with mobilenetv2
#time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt \
#--output_dir ./myFlickr30k_mobile_ep100 --epochs 100 --gpu --encoder mobilenetv2 | tee train_100_myFlickr30k_mobile.log

# For Myanmar Flickr30k (Complete version) with resnext101
#time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt \
#--output_dir ./myFlickr30k_resnext101_ep100 --epochs 100 --gpu --encoder resnext101 | tee train_100_myFlickr30k_resnext101.log

# For Myanmar Flickr30k (Complete version) with resnet50
#time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt \
#--output_dir ./myFlickr30k_resnet50 --gpu --encoder resnet50 | tee train_100_myFlickr30k_resnet50.log

# For Myanmar Flickr30k (Complete version) with resnet101
#time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt \
#--output_dir ./myFlickr30k_resnet101 --gpu --encoder resnet101 | tee train_100_myFlickr30k_resnet101.log

# For Myanmar Flickr30k (Complete version) with resnet152
#time python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt \
#--output_dir ./myFlickr30k_resnet152 --gpu --encoder resnet152 | tee train_100_myFlickr30k_resnet152.log

