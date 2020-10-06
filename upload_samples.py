from azureml.core import Workspace
ws = Workspace.from_config()
datastore = ws.get_default_datastore()
datastore.upload(src_dir='./dashcam_samples', target_path='dashcam_samples', overwrite=True)
