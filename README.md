# sd_image_to_prompts
Kaggle competition: Stable Diffusion images to prompts - https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts

**100th place** solution encoder training repository

# Download data
Download dataset
0. Make dir where to store data
```bash
mkdir /path/to/save/data
mkdir /path/to/save/data/images
```
1. Download metadata
```bash
bash data/download_metadata.sh /path/to/save/data
```
2. Download images
```bash
bash data/download_images.sh ./data/images_datasets_kaggle.txt /path/to/save/data/images
```


# Run training 
```bash
python exp.py
```
