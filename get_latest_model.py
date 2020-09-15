import os

def get_latest_model(prefix="", path=""):
    latest_epoch = 0

    for filename in os.listdir(path):
        if filename.startswith(prefix):
            split = filename.split("-")
            epoch = int(split[2])

            #if we've encountered a later epoch
            if epoch > latest_epoch:
                latest_epoch = epoch

    filepath = path + prefix + "epoch-" + str(latest_epoch)
    if os.path.exists(filepath):
        return {
            "filepath": filepath,
            "latest_epoch": latest_epoch,
        }

    return {
        "filepath": "",
        "latest_epoch": -1,
    }
