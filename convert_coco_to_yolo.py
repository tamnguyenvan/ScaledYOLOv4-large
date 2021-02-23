"""
"""
import os
import shutil
import argparse
import random

import numpy as np
from pycocotools.coco import COCO


random.seed(42)


def load_data(image_dir, annotation_file):
    """Load data from dataset by pycocotools. This tools can be download from
    "http://mscoco.org/dataset/#download"
    Args:
        image_dir: directories of coco images
        annotation_file: file path of coco annotations file
    Return:
        coco_data: list of dictionary format information of each image
    """
    coco = COCO(annotation_file)
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()

    random.shuffle(img_ids)
    coco_data = {}
    num_images = len(img_ids)
    for index, img_id in enumerate(img_ids):
        if index % 100 == 0:
            print("Reading images: %d / %d "%(index, num_images))

        img_info = {}
        bboxes = []
        labels = []

        img_info = coco.loadImgs(img_id)[0]
        height = img_info['height']
        width = img_info['width']
        
        coco_data[img_id] = {
            'filename': img_info['file_name'],
            'width': width,
            'height': height
        }

        ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            bboxes_data = ann['bbox']
            bboxes_data = [bboxes_data[0]/float(width),
                           bboxes_data[1]/float(height),
                           bboxes_data[2]/float(width),
                           bboxes_data[3]/float(height)]
            bboxes.append(bboxes_data)
            labels.append(ann['category_id']-1)

        coco_data[img_id]['bboxes'] = bboxes
        coco_data[img_id]['labels'] = labels

    return coco_data


def create_dataset(image_src, out_dir, data, training=True):
    """
    """
    if training:
        image_dir = os.path.join(out_dir, 'images', 'train')
        label_dir = os.path.join(out_dir, 'labels', 'train')
    else:
        image_dir = os.path.join(out_dir, 'images', 'val')
        label_dir = os.path.join(out_dir, 'labels', 'val')
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    for img_id, img_data in data.items():
        img_path = os.path.join(image_src, img_data['filename'])
        img_out_path = os.path.join(image_dir, img_data['filename'])
        label_path = os.path.join(label_dir, img_data['filename'][:-4] + '.txt')
        shutil.move(img_path, img_out_path)
        with open(label_path, 'wt') as f:
            for bbox, label in zip(img_data['bboxes'], img_data['labels']):
                f.write(' '.join(map(str, [label] + bbox)) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Data dir')
    parser.add_argument('--ann_file', type=str, help='Path to annotation file')
    parser.add_argument('--out_dir', type=str, help='Output dir')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    print(args)
    
    dataset = 'train' if args.train else 'val'
    print(f'Loading {dataset} data')
    data = load_data(args.image_dir, args.ann_file)
    print(f'Creating {dataset} dataset')
    create_dataset(args.image_dir, args.out_dir, data, training=args.train)
    print('Done!')
