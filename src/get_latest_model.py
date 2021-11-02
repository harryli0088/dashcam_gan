# run this file to locate the latest model by parsing the names of the models and choosing the largest epoch

import os

def get_latest_model(prefix="", path=""):
    latest_epoch = 0

    for filename in os.listdir(path): # search through the models in the path
        if filename.startswith(prefix): # if this has a matching prefix
            # parse the model name
            split = filename.split("-")
            epoch = int(split[2])

            if epoch > latest_epoch: #if we've encountered a later epoch
                latest_epoch = epoch # set the new latest epoch

    filepath = path + prefix + "epoch-" + str(latest_epoch) # build the full path to the file
    if os.path.exists(filepath): # if the file path is valid
        return { # return info about the latest model
            "filepath": filepath,
            "latest_epoch": latest_epoch,
        }

    return { # else no models exists
        "filepath": "",
        "latest_epoch": -1,
    }
