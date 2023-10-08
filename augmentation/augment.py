import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

passes = 2
max_rotation_deg = 360.0
max_zoom_pct = 10.0
max_saturation_pct = 10.0
max_brightness_pct = 20.0
max_blur_radius = 1.0
max_contrast_pct = 50.0
input_path = 'input'
output_path = 'output'

def rotate_image_square(im, deg):
    nearest_multiple_of_90 = int(round(deg / 90.0)) * 90
    return im.rotate(nearest_multiple_of_90, expand=True)

# Create the output folder if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.makedirs(output_path)

total_files = 0

# Copy the original folders and their files to the output folder
for folder in os.listdir(input_path):
    os.makedirs(os.path.join(output_path, folder))
    for file in os.listdir(os.path.join(input_path, folder)):
        shutil.copy(os.path.join(input_path, folder, file), os.path.join(output_path, folder, file))
        total_files += 1

# Perform the augmentation passes
for i in range(passes):
    print('Augmentation pass ' + str(i + 1) + ' of ' + str(passes))
    with tqdm(total=total_files) as pbar:
        for folder in os.listdir(input_path):
            for file in os.listdir(os.path.join(input_path, folder)):
                # Create the augmented file name
                file_name = os.path.splitext(file)[0]
                file_ext = os.path.splitext(file)[1]
                augmented_file_name = file_name + '_augmented_' + str(i + 1) + file_ext

                # Generate random augmentation parameters
                rotation = random.uniform(0.0, max_rotation_deg)
                zoom = random.uniform(0.0, max_zoom_pct / 100.0)
                blur = random.uniform(0.0, max_blur_radius)
                saturation = random.uniform(-max_saturation_pct / 100.0, max_saturation_pct / 100.0)
                contrast = random.uniform(-max_contrast_pct / 100.0, max_contrast_pct / 100.0)
                brightness = random.uniform(-max_brightness_pct / 100.0, max_brightness_pct / 100.0)

                image = Image.open(os.path.join(input_path, folder, file))
                # Rotate and zoom so that the image is still fully contained in the frame without black borders
                # image = image.rotate(rotation, expand=False)
                image = rotate_image_square(image, rotation)
                # Blur
                image = image.filter(ImageFilter.GaussianBlur(blur))
                # Saturation
                image = ImageEnhance.Color(image).enhance(1.0 + saturation)
                # Brightness and contrast
                image = ImageEnhance.Brightness(image).enhance(1.0 + brightness)
                image = ImageEnhance.Contrast(image).enhance(1.0 + contrast)
                # Zoom
                image = image.resize((int(image.width * (1.0 + zoom)), int(image.height * (1.0 + zoom))))            
                # Save the augmented image
                image.save(os.path.join(output_path, folder, augmented_file_name))
                pbar.update(1)
