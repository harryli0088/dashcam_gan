# run this file to run the dashcam in azure

from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

if __name__ == "__main__":
    ws = Workspace.from_config()

    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'image_data_train'))
    # models = Dataset.File.from_files(path=(datastore, 'dashcam_model'))
    # samples = Dataset.File.from_files(path=(datastore, 'dashcam_samples'))

    experiment = Experiment(workspace=ws, name='day1-experiment-data')

    config = ScriptRunConfig(
        source_directory='./src',
        script='dashcam_gan.py',
        compute_target='cpu-cluster',
        arguments=[
            '--data_path', dataset.as_named_input('input').as_mount(),
            # '--dashcam_model', models.as_named_input('models').as_mount(),
            # '--dashcam_samples', samples.as_named_input('samples').as_mount()
            ]
        )

    # set up pytorch environment
    env = Environment.from_conda_specification(name='pytorch-env',file_path='.azureml/pytorch-env.yml')
    config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print("Submitted to an Azure Machine Learning compute cluster. Click on the link below")
    print("")
    print(aml_url)
