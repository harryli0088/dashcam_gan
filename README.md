# dashcam_gan

My attempt at creating a Generative Adversarial Network using my car dashcam video data.



## Setting up
Load your image data into ```data/image_data_train/1```

## Running locally
Run ```src/dashcam_gan.py```


## Running on Microsft Azure
Followed this tutorial series to run the script on the Microsoft Azure: https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-setup-local

1. Fill out ```create_workplace_template.py``` with your account credentials and run it
2. Run ```check_images.py```
3. Run ```upload_data.py```
4. Run ```run_dashcam_gan.py```
