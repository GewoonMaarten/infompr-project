from pathlib import Path
from tqdm import tqdm
import argparse
import json
import os
import pandas as pd
import shutil

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

  for i in tqdm(range(sample_size), unit='Images'):
    stem = df.iloc[i].id
    shutil.copyfile(image_paths[stem], Path(dest, 'images', f'{stem}.jpg'))

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
    '-z',
    dest='zip',
    action='store_true',
    help='should the minidataset be zipped or not')
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
  create_dir(f'dist/mini_dataset_{args.dataset}/images')

  image_paths = get_image_paths(obj['public_dataset']['images_dir'])
  create_mini_dataset(
    args.dataset,
    obj['public_dataset'][f'multimodal_{args.dataset}'], 
    args.samples,
    image_paths,
    dest)
  
  if(args.zip):
    shutil.make_archive(dest, 'zip', dest)