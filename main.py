import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import shutil
import os

IMAGE_DIRECTORY = 'C:\\Users\\petr.matl\\PalletsImages'

pallets = []

for root, dirs, files in os.walk(IMAGE_DIRECTORY):
    for file in files:
        if file.endswith('.bmp'):
            img = Image.open(os.path.join(root, file))
            resized_image = ImageOps.fit(img, (128, 160), Image.ANTIALIAS)
            #resized_image.save(os.path.join(IMAGE_DIRECTORY, '128x160', file.replace('.bmp', '.jpg')), 'jpeg')
            image_info = file[:-4].split('_')
            pallets.append(np.concatenate([image_info[0], image_info[1]], np.asarray(resized_image).flatten()))
            #images_labels.append([image_info[0], image_info[1]])
            break
    
np.savetxt(os.path.join(IMAGE_DIRECTORY, 'pallets.csv'), pallets, delimiter=",", fmt='%s')
#np.savetxt(os.path.join(IMAGE_DIRECTORY, 'images_labels.csv'), images_labels, delimiter=",", fmt='%s')
    
#print(np.asarray(img).shape)
#print(np.asarray(new_img).shape)