import argparse
from pathlib import Path
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='run_local.py',
        description='')
    parser.add_argument(
        '--trainer',
        choices=['dual', 'image', 'text'],
        required=True,
        help='which dataset should be used to create the minidataset')
    args = parser.parse_args()

    try:
        path = Path('config.json')
        with open(path, 'r') as f:
            config = json.load(f)
    except IOError:
        print('config.json does not exist!')
        print('View the README.md on how to create one.')
        raise

    print('config.json:')
    print(json.dumps(config, indent = 3)) 
    print('Check if the config is correct!')
    input("Press Enter to continue...")

    if args.trainer == 'image':
        import trainers.efficientnet
    elif args.trainer == 'text':
        import trainers.title
    elif args.trainer == 'dual':
        import trainers.dual
    else:
        raise ValueError('Trainer does not have a valid value.')
