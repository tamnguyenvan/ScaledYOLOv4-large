# ScaledYOLOv4-large

## Installation
Install `torch` and other dependencies in `requirements.txt`.

Download COCO dataset. Put it and your dataset into `data` folder as follows.
```
data
| -- train2017.zip
| -- val2017.zip
| -- data.zip
```

Unzip them then run data preparing scripts. First, run the following command to convert COCO format to YOLO format.
```
python convert_coco_to_yolo.py --image_dir ./data/val2017 --ann_file ./data/annotations/instances_val2017.json --out_dir ./data
```

Merge your data with COCO.
```
python merge_data.py --label_dir ./data/data/labels/train2017 --split 0.1 --coco_image_dir ./data/images/
```

## Training
Train the model.
```
python train.py --cfg ./data/cocoplus.yaml --hyp ./data/hyp.finetune.yaml --batch-size 1
```