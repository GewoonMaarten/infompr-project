from pathlib import Path
from tqdm import tqdm
import argparse
import json
import os
import pandas as pd
import zipfile

def create_dir(name):
  root = Path(os.path.realpath(__file__)).parent.parent
  path = Path(root, name)
  path.mkdir(parents=True, exist_ok=True)
  return path

def get_image_paths(path):
  paths = Path(path).glob('*.jpg')
  return {x.stem:x for x in paths}

def create_mini_dataset(name, path, sample_size, image_paths, dest):
  df = pd.read_csv(path, sep='\t', header=0)
  df = df.sample(n=sample_size, replace=False)
  zipFile = zipfile.ZipFile(Path('dist', f'mini_dataset_{name}.zip'), 'w')

  for i in tqdm(range(sample_size), unit='Images'):
    stem = df.iloc[i].id
    zipFile.write(image_paths[stem], arcname=f'images/{stem}.jpg')

  df.to_csv(Path(dest, f'mini_dataset_{name}.tsv'), sep='\t', index=False)
  zipFile.write(
    Path(dest, f'mini_dataset_{name}.tsv'), 
    arcname=f'mini_dataset_{name}.tsv')
  zipFile.close()

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

  obj = None
  with open('config.json', 'r') as f:
    config = f.read()
    obj = json.loads(config)
  
  dest = create_dir(f'dist/mini_dataset_{args.dataset}')

  image_paths = get_image_paths(obj['public_dataset']['images_dir'])
  create_mini_dataset(
    args.dataset,
    obj['public_dataset'][f'multimodal_{args.dataset}'], 
    args.samples,
    image_paths,
    dest)
