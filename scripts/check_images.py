from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd

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
  obj = None
  with open('config.json', 'r') as f:
    config = f.read()
    obj = json.loads(config)

  image_ids = get_image_ids(obj['public_dataset']['images_dir'])

  for p in [
    obj['public_dataset']['multimodal_test'],
    obj['public_dataset']['multimodal_train'],
    obj['public_dataset']['multimodal_validate']
  ]:
    check_tsv(p, image_ids)