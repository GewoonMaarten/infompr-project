from pathlib import Path
from tqdm import tqdm
import argparse
import os
import pandas as pd
import shutil
import urllib.request
from PIL import Image
import math

try:
    from utils.config import (
        dataset_test_path,
        dataset_train_path,
        dataset_validate_path,
        dataset_images_path)
except ImportError:
    print('Run script as: python -m scripts.create_mini_dataset\n')
    raise


def create_dir(name):
    root = Path(os.path.realpath(__file__)).parent.parent
    path = Path(root, name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_mini_dataset(name, path, sample_size, dest):
    df = pd.read_csv(path, sep='\t', header=0)
    # TODO Tom maybe we can find a way to retreive more samples if images are corrupt/not found
    df_one = df[df['2_way_label'] == 1] \
        .sample(n=math.floor(sample_size / 2), replace=False)
    df_zero = df[df['2_way_label'] == 0] \
        .sample(n=math.floor(sample_size / 2), replace=False)
    df_new = pd.concat([df_one, df_zero], axis=0) 

    df_new['2_way_label'] = pd.Categorical(df_new['2_way_label'])
    print(df_new['2_way_label'].value_counts())

    # If the images folder does not exist
    # we create it as a cache folder
    if not Path(dataset_images_path).exists():
        create_dir(dataset_images_path)

    for _, row in tqdm(df_new.iterrows(), total=len(df_new), unit='Images'):
        id = row['id']
        img_src = Path(dataset_images_path, f'{id}.jpg')
        img_dest = Path(dest, 'images', f'{id}.jpg')

        # try download & open the image
        if not img_src.exists():
            try:
                urllib.request.urlretrieve(row['image_url'], img_src)
                try:
                    Image.open(str(img_src))
                except IOError:
                    img_src.unlink()
                    continue
            except urllib.error.HTTPError:
                continue

        shutil.copy(img_src, img_dest)

    df.to_csv(Path(dest, f'mini_dataset_{name}.tsv'), sep='\t', index=False)


if __name__ == "__main__":
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

    print('!!!IMPORTANT!!!')
    print('Before using this script makes sure that your dataset is clean!')
    print('Run "python -m scripts.fix_dataset" if you have not done so already')
    input("Press Enter to continue...")

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
