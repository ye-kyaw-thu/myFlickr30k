#!/bin/bash

#python3.10 captioner.py --data_dir ./data/flickr30k_images --caption_file captions.txt  --output_dir ./ep60 \
#--resume ./ep50/final_model.ckpt --epochs 10 | tee resume_10.log

#python3.10 captioner.py --data_dir ./data/flickr30k_images --caption_file captions.txt  --output_dir ./ep70 \
#--resume ./ep60/final_model.ckpt --epochs 3 | tee resume_3.log

#python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt  --output_dir ./my_ep100 \
#--resume ./my_ep50/final_model.ckpt --epochs 50 | tee resume_50.log

python3.10 captioner_my.py --data_dir ../data/flickr30k_images --caption_file my_captions.txt  --output_dir ./my_ep200 \
--resume ./my_ep100/final_model.ckpt --epochs 100
