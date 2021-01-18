import argparse

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

    if args.trainer == 'image':
        import trainers.efficientnet
    elif args.trainer == 'text':
        import trainers.bert_roberta
    elif args.trainer == 'dual':
        import trainers.dual
    else:
        raise ValueError('Trainer does not have a valid value.')
