import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

def check_tsv(path, image_ids):
  df = pd.read_csv(path, sep='\t', header=0)
  number_of_rows = df.shape[0]

  print(f'Checking {path}:')
  for i in tqdm(range(number_of_rows), unit=' Images'):
    if not df.iloc[i].id in image_ids:
      print(f'image {df.iloc[i].id} not found!')

def get_image_ids(path):
  paths = Path(path).glob('*.jpg')
  return [x.stem for x in paths]

if __name__ == "__main__":
  obj = None
  with open('config.json', 'r') as f:
    config = f.read()
    obj = json.loads(config)

  multimodal_test_path = obj['public_dataset']['multimodal_test']
  multimodal_train_path = obj['public_dataset']['multimodal_train']
  multimodal_validate_path = obj['public_dataset']['multimodal_validate']
  images_dir_path = obj['public_dataset']['images_dir']

  image_ids = get_image_ids(images_dir_path)

  check_tsv(multimodal_test_path, image_ids)
  check_tsv(multimodal_train_path, image_ids)
  check_tsv(multimodal_validate_path, image_ids)