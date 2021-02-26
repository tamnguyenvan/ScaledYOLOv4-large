import os
import glob
import shutil
import logging
import argparse
import random
import tqdm
import numpy as np


logging.basicConfig(filename='merge.log', level=logging.DEBUG)


def main():
    # Load image paths
    label_paths = glob.glob(os.path.join(args.label_dir, '*'))
    image_paths = [label_path.replace('labels', 'images').replace('.txt', '.png') for label_path in label_paths]
    print(f'Found {len(image_paths)} images')
    
    split = args.split
    dataset = args.dataset
    if isinstance(split, float) and split > 0:
        random.seed(42)
        paths = list(zip(image_paths, label_paths))
        random.shuffle(paths)
        image_paths, label_paths = zip(*paths)

        num_train = int(len(image_paths) * (1 - split))
        train_image_paths = image_paths[:num_train]
        train_label_paths = label_paths[:num_train]
        val_image_paths = image_paths[num_train:]
        val_label_paths = label_paths[num_train:]
        
        source = {
            'train': {'image': train_image_paths, 'label': train_label_paths},
            'val': {'image': val_image_paths, 'label': val_label_paths},
        }
    else:
        source = {
            dataset: {'image': image_paths, 'label': label_paths},
        }

    class_id = args.begin_class_id
    out_image_dir = args.coco_image_dir
    out_label_dir = out_image_dir.replace('images', 'labels')
    for split, data in source.items():
        print(f'Merging {split}')
        out_split_image_dir = os.path.join(out_image_dir, split)
        out_split_label_dir = os.path.join(out_label_dir, split)
        for image_path, label_path in tqdm.tqdm(zip(data['image'], data['label'])):
            image_filename = os.path.basename(image_path)
            new_image_path = os.path.join(out_split_image_dir, image_filename)
            shutil.copy(image_path, new_image_path)
            logging.info(f'Copied {image_path} to {new_image_path}')
            
            # Modify original class id
            annotation = np.loadtxt(label_path, dtype=np.float32, delimiter=' ')
            if len(annotation.shape) == 1:
                annotation = annotation[None, :]
            annotation[:, 0] += class_id
            annotation = annotation.tolist()
            label_filename = os.path.basename(label_path)
            new_label_path = os.path.join(out_split_label_dir, label_filename)
            with open(new_label_path, 'wt') as f:
                for row in annotation:
                    if int(row[0]) < 0:
                        print(f'Found negative id: {row[0]} in {label_path}')
                        continue
                    row = [int(row[0])] + row[1:]
                    f.write(' '.join(map(str, row)) + '\n')
            logging.info(f'Copied {label_path} to {new_label_path}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir', type=str, help='Path to label dir')
    parser.add_argument('--split', type=float, default=0.1, help='Split percentage')
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'val', 'test'], help='Dataset type')
    parser.add_argument('--coco_image_dir', type=str, help='Path to COCO image dir')
    parser.add_argument('--begin_class_id', type=int, default=80, help='Beginning class id')
    args = parser.parse_args()
    print(args)

    main()