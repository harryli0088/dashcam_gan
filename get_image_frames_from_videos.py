import cv2
import os
import sys

# recursively get all file paths
def get_file_paths(starting_dir):
    file_paths = []
    for root, subdirs, files in os.walk(starting_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append([
                file_path, filename
            ])
    return file_paths

file_paths = get_file_paths("video_data")
print("file_paths",file_paths)

# extract still frames from video files
for file_path in file_paths:
    print("reading frames from",file_path[0])
    vc = cv2.VideoCapture(file_path[0]) # try to open the file as a video
    c=1 # used to name the output image files

    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False

    while rval: # if there is still data to read
        try:
            rval, frame = vc.read() # read the next frame
            cv2.imwrite('data/image_data_train/1/' + file_path[1] + "-" + str(c) + '.jpg',frame) # write the image file
            c = c + 1 # increment the counter
            cv2.waitKey(1)
        except NameError:
            print(NameError)

    vc.release()
