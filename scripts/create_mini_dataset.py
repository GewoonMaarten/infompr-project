from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import os
import pandas as pd

try:
    from utils.config import (
        dataset_test_path,
        dataset_train_path,
        dataset_validate_path,
        dataset_images_path)
except ImportError:
    print('Run script as: python -m scripts.check_images\n')
    raise


def create_dir(name):
    root = Path(os.path.realpath(__file__)).parent.parent
    path = Path(root, name)
    path.mkdir(parents=True, exist_ok=True)
    return path



def create_mini_dataset(name, path, sample_size, dest):
    df = pd.read_csv(path, sep='\t', header=0)

    pbar = tqdm(total=sample_size, unit='Images')
    samples = set()
    while sample_size > 0:
        sample = df.sample()
        index = sample.index[0]
        if index in samples:
            continue

        id = sample.iloc[0].id

        img_src = Path(dataset_images_path, f'{id}.jpg')
        img_dest = Path(dest, 'images', f'{id}.jpg')
        img = cv2.imread(str(img_src))
        if img is None:
            continue
        cv2.imwrite(str(img_dest), img)

        samples.add(index)
        sample_size -= 1
        pbar.update()
    pbar.close()

    df = df[df.index.isin(samples)]
    df.to_csv(Path(dest, f'mini_dataset_{name}.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        dest='samples',
        required=True,
        type=int,
        help='number of samples for the mini dataset')
    parser.add_argument(
        '-d',
        dest='dataset',
        choices=['train', 'test', 'validate'],
        default='train',
        help='which dataset should be used to create the minidataset')
    args = parser.parse_args()

    dest = create_dir('dist')
    create_dir('dist/images')

    dataset_path = None
    if args.dataset == 'test':
        dataset_path = dataset_test_path
    elif args.dataset == 'train':
        dataset_path = dataset_train_path
    elif args.dataset == 'validate':
        dataset_path = dataset_validate_path

    create_mini_dataset(
        args.dataset,
        dataset_path,
        args.samples,
        dest)
