# INFOMPR Project

### Config file structure
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