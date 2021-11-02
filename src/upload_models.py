# run this model to upload existing models to azure
from azureml.core import Workspace
ws = Workspace.from_config()
datastore = ws.get_default_datastore()
datastore.upload(src_dir='./dashcam_model', target_path='dashcam_model', overwrite=True)
