# run this file to move some training files to testing

import os
import random

def move_train_to_test(src_path="",dest_path="", percentage=10):
    threshold = 1 - percentage/100

    for filename in os.listdir(src_path):
        if(random.random() > threshold):
            os.replace(src_path+filename, dest_path+filename)




move_train_to_test('image_data_train/1/','image_data_test/1/', 10)
