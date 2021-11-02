# run this file to pull out still frames out of your videos

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
print("file_paths", file_paths)

SKIP = 4  # save every SKIP-th frame

# extract still frames from video files
for file_path in file_paths:
    print("reading frames from", file_path[0])
    vc = cv2.VideoCapture(file_path[0])  # try to open the file as a video
    file_counter = 1  # used to name the output image files

    rval = False
    if vc.isOpened():
        rval, frame = vc.read()

    skip_counter = 0 # used to track whether we should keep or skip a frame
    while rval:  # if there is still data to read
        try:
            rval, frame = vc.read()  # read the next frame
            if frame is not None:  # if there is frame data
                if skip_counter>=SKIP: # if we should save this frame
                    # write the image file
                    cv2.imwrite('data/image_data_train/1/' +
                                file_path[1] + "-" + str(file_counter) + '.jpg', frame)
                    cv2.waitKey(1)
                    file_counter += 1  # increment the counter
                    skip_counter = 0 # reset i to 0
                else:
                    skip_counter += 1 # increment i
            else:  # else we are probably at the end of the video
                break  # break out of the loop
           
        except NameError:
            print(NameError)

    vc.release()
