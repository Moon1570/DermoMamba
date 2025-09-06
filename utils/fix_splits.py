import os
import glob

# Get actual image files
train_images = glob.glob('data/ISIC2018_proc/train_images/*.png')
val_images = glob.glob('data/ISIC2018_proc/val_images/*.png')

# Extract just the image IDs (without .png)
train_ids = [os.path.basename(f).replace('.png', '') for f in train_images]
val_ids = [os.path.basename(f).replace('.png', '') for f in val_images]

print(f'Found {len(train_ids)} training images')
print(f'Found {len(val_ids)} validation images')

# Write new split files
with open('splits/isic2018_train.txt', 'w') as f:
    for img_id in sorted(train_ids):
        f.write(img_id + '\n')

with open('splits/isic2018_val.txt', 'w') as f:
    for img_id in sorted(val_ids):
        f.write(img_id + '\n')

print('Updated split files based on actual images')
print('First 5 train IDs:', train_ids[:5])
print('First 5 val IDs:', val_ids[:5])
