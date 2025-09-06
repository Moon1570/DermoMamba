import os
import glob
import random

# Get all training images that have corresponding masks
train_images = glob.glob('data/ISIC2018_proc/train_images/*.png')
train_masks = glob.glob('data/ISIC2018_proc/train_masks/*_segmentation.png')

# Extract image IDs
train_image_ids = [os.path.basename(f).replace('.png', '') for f in train_images]
mask_ids = [os.path.basename(f).replace('_segmentation.png', '') for f in train_masks]

# Find images that have corresponding masks
valid_ids = [img_id for img_id in train_image_ids if img_id in mask_ids]

print(f'Total training images: {len(train_image_ids)}')
print(f'Total masks: {len(mask_ids)}')
print(f'Valid image-mask pairs: {len(valid_ids)}')

# Split into train/val (80/20)
random.seed(42)
random.shuffle(valid_ids)

split_idx = int(0.8 * len(valid_ids))
train_ids = valid_ids[:split_idx]
val_ids = valid_ids[split_idx:]

print(f'New train split: {len(train_ids)} images')
print(f'New val split: {len(val_ids)} images')

# Write new split files
with open('splits/isic2018_train.txt', 'w') as f:
    for img_id in sorted(train_ids):
        f.write(img_id + '\n')

with open('splits/isic2018_val.txt', 'w') as f:
    for img_id in sorted(val_ids):
        f.write(img_id + '\n')

print('Updated split files with valid image-mask pairs')
print(f'Sample train IDs: {train_ids[:5]}')
print(f'Sample val IDs: {val_ids[:5]}')
