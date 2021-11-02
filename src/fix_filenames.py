# run this file to tweak the naming convention of your files
import os

def fix_filenames(path):
    for filename in os.listdir(path):
        split = filename.split(".")
        if(len(split)==3 and split[2]=="jpg"):
            print(split[0] + "-" + split[1] + "." + split[2])
            os.rename(path+filename, path+split[0] + "-" + split[1] + "." + split[2])


fix_filenames("./image_data/train/")
