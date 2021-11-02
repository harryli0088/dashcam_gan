# run this file to validate the dimensions of your training images
import os
from PIL import Image

def check_images(path):
    for filename in os.listdir(path):
        file_path = path + filename
        file_size = os.path.getsize(file_path)
        im = Image.open(file_path)
        dimensions = im.size
        if dimensions[0]!=1280 and dimensions[1]!=960:
            print(im.size, file_size, file_path)


check_images("./data/image_data_train/1/")
