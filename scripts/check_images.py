from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import logging
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


def check_tsv(path, image_ids, thorough):
    logging.info(f'Checking {path}:')

    df = pd.read_csv(path, sep='\t', header=0)
    number_of_rows = df.shape[0]

    imgs_not_found = []
    imgs_not_loaded = []
    for i in tqdm(range(number_of_rows), unit='Images'):
        id = df.iloc[i].id

        if not id in image_ids:
            imgs_not_found.append(id)
            continue

        if thorough:
            path = Path(dataset_images_path, f'{id}.jpg')
            img = cv2.imread(str(path))
            if img is None:
                imgs_not_loaded.append(id)

    if not imgs_not_found and not imgs_not_loaded:
        logging.info('No missing or incorrect images found')

    if imgs_not_found:
        logging.error('Images not found:')
        logging.error('\n'.join(imgs_not_found))

    if thorough and imgs_not_loaded:
        logging.error('Images not loaded:')
        logging.error('\n'.join(imgs_not_loaded))


def get_image_ids(path):
    paths = Path(path).glob('*.jpg')
    return set([x.stem for x in paths])


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s]: %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        dest='thorough',
        required=False,
        action='store_true',
        help='Checks the images thoroughly by loading them.')
    args = parser.parse_args()

    image_ids = get_image_ids(dataset_images_path)

    for p in [dataset_test_path, dataset_train_path, dataset_validate_path]:
        check_tsv(p, image_ids, args.thorough)
