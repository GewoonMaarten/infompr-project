# INFOMPR Project

## Links
* [Fakeddit Github](https://github.com/entitize/Fakeddit)

Dataset:
* [Text + metadata](https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm?usp=sharing)
* [Images](https://drive.google.com/file/d/1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b/view?usp=sharing)
* [Comments](https://drive.google.com/drive/folders/150sL4SNi5zFK8nmllv5prWbn0LyvLzvo?usp=sharing)

## Config file structure
The config file is used to store the paths to the datasets. 
It needs to be called `config.json` and needs to be stored in the root of the project.
It needs to have the following structure (keys with the `_` prefix are optional):
```json
{
  "public_dataset": 
  {
    "multimodal_test": "path_to_multimodal_test_public.tsv",
    "multimodal_train": "path_to_multimodal_train.tsv",
    "multimodal_validate": "path_to_multimodal_validate.tsv",
    "images_dir": "path_to_public_image_set_dir"
  },
  "epochs": 0,
  "batch_size": 0,
  "text_config": {
    "max_length": 0
  },
  "img_config": {
    "img_width": 0,
    "img_height": 0
  },
  "_teams_webhook_url": "webhook_url"
}
```
This file is then loaded in `utils.config.py` to be made available for the whole project.

These are the settings used in the final models:
```json
{
  "epochs": 10,
  "batch_size": 10,
  "text_config": {
    "max_length": 128
  },
  "img_config": {
    "img_width": 380,
    "img_height": 380
  }
}
```

## Scripts
Below are some scripts that can help you with training of the models.
They need to be executed from the root directory of the project.

* `scripts.check_images.py` can be used to check if all images exist and can be loaded.
  ```
  > python -m scripts.check_images
  usage: check_images.py [-h] [-t]

  optional arguments:
    -h, --help  show this help message and exit
    -t          checks the images thoroughly by loading them
  ```

* `scripts.create_mini_dataset.py` can be used to create a mini dataset from the larger one.
  ```
  > python -m scripts.create_mini_dataset
  usage: create_mini_dataset.py [-h] -n SAMPLES [-d {train,test,validate}]

  optional arguments:
    -h, --help               show this help message and exit
    -n SAMPLES               number of samples for the mini dataset
    -d {train,test,validate} which dataset should be used to create the minidataset
  ```

* `scripts.fix_dataset.py` can be used to remove invalid images from the dataset.
  ```
  > python -m scripts.fix_dataset
  ```
