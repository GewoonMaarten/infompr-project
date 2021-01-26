import json
from pathlib import Path

try:
    path = Path(Path(__file__).parent.parent.absolute(), 'config.json')
    with open(path, 'r') as f:
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

training_epochs = config['epochs']
training_batch_size = config['batch_size']

text_max_length = config['text_config']['max_length']
text_use_bert = config['text_config']['use_bert']
print(text_use_bert)

img_width = config['img_config']['img_width']
img_height = config['img_config']['img_height']
img_size = (img_width, img_height)

# Optional config
try:
    teams_webhook_url = config['_teams_webhook_url']
except KeyError:
    pass
