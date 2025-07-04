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

There are 7 experiment folders under the `exp_1/` directory. These are named according to the image feature extractor used:

1. `mobilenetv2_ep100`  
2. `resnet50_ep100`  
3. `resnet101_ep100`  
4. `resnet152_ep100`  
5. `resnext50_ep100`  
6. `resnext101_ep100`  
7. `vgg16_ep100`
