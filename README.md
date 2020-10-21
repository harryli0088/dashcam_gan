# dashcam_gan

My attempt at a Generative Adversarial Network using my car dashcam video data. I had a some dashcam video footage from which I pulled ~26,000 frames of images to use as training data. I trained the model using Microsoft Azure's Machine Learning service.

Example sample image:
![Example sample image](example_sample_image.jpg)

Example Output Images from Epoch 51
![Example output 1](outputs/sample_51-1.png)
![Example output 2](outputs/sample_51-2.png)

Discriminator Loss
![Discriminator Loss](outputs/d_loss.png)

Generator Loss
![Generator Loss](outputs/g_loss.png)

## Setting up
Load your image data into ```data/image_data_train/1```

## Running locally
Run ```src/dashcam_gan.py```


## Running on Microsft Azure
I followed this tutorial series to run the script on the Microsoft Azure: https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-setup-local

1. Fill out ```create_workplace_template.py``` with your account credentials and run it
2. Run ```check_images.py```
3. Run ```upload_data.py```
4. Run ```run_dashcam_gan.py```
