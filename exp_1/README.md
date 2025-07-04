# File/Folder Description

## Python Codes

1. **`captioner_my.py`** – Python script for training and testing the image-to-caption model.  
2. **`copy_images.py`** – Copies sample images based on entries in the `predictions.txt` file.  
3. **`mk_html.py`** – Converts the `predictions.txt` file into an HTML format for human-friendly inspection.

## Shell Scripts

1. **`train.sh`** – Shell script for training/testing the model using 7 types of image features.  
2. **`resume_train.sh`** – Shell script for resuming training from the last saved epoch.  
3. **`mk_7html.sh`** – Shell script that converts `predictions.txt` files into HTML format for all 7 experiments.

## Experiment Folders

Although the actual training output folders contain large model files (as shown below), only the result files have been uploaded due to GitHub’s file size limitations:

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ tree myFlickr30k_vgg_ep100/
myFlickr30k_vgg_ep100/
├── detailed_predictions.json
├── final_model.ckpt
├── lightning_logs
│   └── version_0
│       ├── checkpoints
│       │   └── epoch=99-step=9400.ckpt
│       ├── events.out.tfevents.1751299502.lst-hpc3090.2045453.0
│       ├── events.out.tfevents.1751318529.lst-hpc3090.2045453.1
│       └── hparams.yaml
├── predictions.html
├── predictions.txt
├── test_loss.png
└── train_val_loss.png

4 directories, 10 files
```

```
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$ ll -h -R ./myFlickr30k_vgg_ep100/*
-rw-rw-r-- 1 ye ye 4.9M Jul  1 04:24 ./myFlickr30k_vgg_ep100/detailed_predictions.json
-rw------- 1 ye ye 363M Jul  1 04:22 ./myFlickr30k_vgg_ep100/final_model.ckpt
-rw-rw-r-- 1 ye ye  18M Jul  5 01:27 ./myFlickr30k_vgg_ep100/predictions.html
-rw-rw-r-- 1 ye ye 151K Jul  1 04:24 ./myFlickr30k_vgg_ep100/predictions.txt
-rw-rw-r-- 1 ye ye  40K Jul  1 04:24 ./myFlickr30k_vgg_ep100/test_loss.png
-rw-rw-r-- 1 ye ye  31K Jul  1 04:24 ./myFlickr30k_vgg_ep100/train_val_loss.png

./myFlickr30k_vgg_ep100/lightning_logs:
total 12K
drwxrwxr-x 3 ye ye 4.0K Jun 30 23:05 ./
drwxrwxr-x 3 ye ye 4.0K Jul  3 06:18 ../
drwxr-xr-x 3 ye ye 4.0K Jul  1 04:22 version_0/

./myFlickr30k_vgg_ep100/lightning_logs/version_0:
total 84K
drwxr-xr-x 3 ye ye 4.0K Jul  1 04:22 ./
drwxrwxr-x 3 ye ye 4.0K Jun 30 23:05 ../
drwxrwxr-x 2 ye ye 4.0K Jul  1 04:22 checkpoints/
-rw-rw-r-- 1 ye ye  57K Jul  1 04:22 events.out.tfevents.1751299502.lst-hpc3090.2045453.0
-rw-rw-r-- 1 ye ye 1.5K Jul  1 04:24 events.out.tfevents.1751318529.lst-hpc3090.2045453.1
-rw-rw-r-- 1 ye ye  462 Jun 30 23:05 hparams.yaml

./myFlickr30k_vgg_ep100/lightning_logs/version_0/checkpoints:
total 363M
drwxrwxr-x 2 ye ye 4.0K Jul  1 04:22  ./
drwxr-xr-x 3 ye ye 4.0K Jul  1 04:22  ../
-rw------- 1 ye ye 363M Jul  1 04:22 'epoch=99-step=9400.ckpt'
(pytorch_py3.10) ye@lst-hpc3090:~/intern3/exp/captioning/my$
```

There are 7 experiment folders under the `exp_1/` directory. These are named according to the image feature extractor used:

1. `mobilenetv2_ep100`  
2. `resnet50_ep100`  
3. `resnet101_ep100`  
4. `resnet152_ep100`  
5. `resnext50_ep100`  
6. `resnext101_ep100`  
7. `vgg16_ep100`

## To Do

- Manually check the translated Myanmar sentences  
- Update `captioner_my.py` (currently supports only RNN and LSTM)  
- Experiment with word-segmented Myanmar captions  

