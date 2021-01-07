import json

config = None
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except IOError:
    print('config.json does not exist!')
    print('View the README.md on how to create one.')
    raise

# Required config
dataset_test_path = config['public_dataset']['multimodal_test']
dataset_train_path = config['public_dataset']['multimodal_train']
dataset_validate_path = config['public_dataset']['multimodal_validate']
dataset_images_path = config['public_dataset']['images_dir']

# Optional config
try:
    teams_webhook_url = config['_teams_webhook_url']
except KeyError:
    pass
