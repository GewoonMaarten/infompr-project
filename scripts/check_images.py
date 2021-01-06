from pathlib import Path
from tqdm import tqdm
import pandas as pd

try:
    from utils.config import (
        dataset_test_path,
        dataset_train_path,
        dataset_validate_path,
        dataset_images_path)
except ImportError:
    print('Run script as: python -m script.check_images\n')
    raise


def check_tsv(path, image_ids):
    print(f'Checking {path}:')

    df = pd.read_csv(path, sep='\t', header=0)
    number_of_rows = df.shape[0]

    for i in tqdm(range(number_of_rows), unit='Images'):
        if not df.iloc[i].id in image_ids:
            print(f'Image \'{df.iloc[i].id}\' not found!')


def get_image_ids(path):
    paths = Path(path).glob('*.jpg')
    return set([x.stem for x in paths])


if __name__ == "__main__":
    image_ids = get_image_ids(dataset_images_path)

    for p in [dataset_test_path, dataset_train_path, dataset_validate_path]:
        check_tsv(p, image_ids)
