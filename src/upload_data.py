# run this file to upload all the training data to azure
from azureml.core import Workspace
ws = Workspace.from_config()
datastore = ws.get_default_datastore()
datastore.upload(src_dir='./data', target_path='image_data_train', overwrite=True)
