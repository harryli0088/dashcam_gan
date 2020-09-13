import os
import sys

def get_file_paths(starting_dir):
    file_paths = []
    for root, subdirs, files in os.walk(starting_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append([
                file_path, filename
            ])
    return file_paths

file_paths = get_file_paths("video-data")

import cv2
for file_path in file_paths:
    print("reading frames from",file_path[0])
    vc = cv2.VideoCapture(file_path[0])
    c=1

    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False

    while rval:
        try:
            rval, frame = vc.read()
            cv2.imwrite('data/' + file_path[1] + "-" + str(c) + '.jpg',frame)
            c = c + 1
            cv2.waitKey(1)
        except NameError:
            print(NameError)
        except:
            print("test")
    vc.release()
