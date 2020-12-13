# INFOMPR Project

## Links
* [Fakeddit Github](https://github.com/entitize/Fakeddit)

Dataset:
* [Text + metadata](https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm?usp=sharing)
* [Images](https://drive.google.com/file/d/1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b/view?usp=sharing)
* [Comments](https://drive.google.com/drive/folders/150sL4SNi5zFK8nmllv5prWbn0LyvLzvo?usp=sharing)

## Config file structure
The config file is used to store the paths to the datasets.
It has the following structure:
```json
{
  "public_dataset": 
  {
    "multimodal_test": "path_to_multimodal_test_public.tsv",
    "multimodal_train": "path_to_multimodal_train.tsv",
    "multimodal_validate": "path_to_multimodal_validate.tsv",
    "images_dir": "path_to_public_image_set_dir"
  }
}
```