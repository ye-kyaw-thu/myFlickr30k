# File/Folder Description

## Python Codes

1. captioner_my.py (training/testing image2caption python code)
2. copy_images.py (copying sample images based on predictions.txt file)
3. mk_html.py (conversion of predictions.txt file into HTML format file for human checking)
   
## Shell Scripts

1. train.sh (shell script for training/testing with 7 types of image feature)
2. resume_train.sh (shell script for resume training from last trained epoch)
3. mk_7html.sh (shell script of converting predictions.txt file into html for all 7 experiments)

## Experiment Folders

Though actual training output folder will contain model files as shown below, I only uploaded result files because of filesize limitation of GitHub:

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



1. mobilenetv2_ep100
2. resnet50_ep100
3. resnet101_ep100
4. resnet152_ep100
5. resnext50_ep100
6. resnext101_ep100
7. vgg16_ep100
