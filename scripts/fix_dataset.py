from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

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
        try:
            img = tf.io.read_file(str(img_path))
            img = tf.image.decode_jpeg(img, channels=3)
        except:
            indexes_to_drop.append(index)

    df = df.drop(indexes_to_drop)
    df.to_csv(path, sep='\t', index=False)

    print(f'removed {len(indexes_to_drop)} images')


if __name__ == '__main__':
    for path in [dataset_test_path, dataset_train_path, dataset_validate_path]:
        fix_dataset(path)
