import os
from PIL import Image

def check_images(path):
    for filename in os.listdir(path):
        file_path = path + filename
        file_size = os.path.getsize(file_path)
        im = Image.open(file_path)
        dimensions = im.size
        if dimensions[0]!=1920 and dimensions[1]!=1080:
            print(im.size, file_size, file_path)


check_images("./image_data_train/1/")
check_images("./image_data_test/1/")
