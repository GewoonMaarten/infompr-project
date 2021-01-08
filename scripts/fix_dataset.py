import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

try:
    from utils.config import (
        dataset_test_path,
        dataset_train_path,
        dataset_validate_path,
        dataset_images_path)
except ImportError:
    print('Run script as: python -m scripts.fix_dataset\n')
    raise


def fix_dataset(path):
    print(f'Fixing {path}:')

    df = pd.read_csv(path, sep='\t', header=0)
    indexes_to_drop = []
    for index, row in tqdm(df.iterrows(), total=len(df), unit='Images'):
        id = row['id']
        img_path = Path(dataset_images_path, f'{id}.jpg')
        img = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if img is None:
            indexes_to_drop.append(index)

    df = df.drop(indexes_to_drop)
    df.to_csv(path, sep='\t', index=False)

if __name__ == '__main__':
    for path in [dataset_test_path, dataset_train_path, dataset_validate_path]:
        fix_dataset(path)
