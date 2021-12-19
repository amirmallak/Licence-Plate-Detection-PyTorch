# Licence-Plate-Detection-PyTorch

Training a model to find the area in the image that corresponds to a number plate.
The database provides the bounding boxes of the ground truth location of the plate. 

A straightforward but very inefficient way to solve this problem is to create a training/validation set of images for two classes: 
plate and background (not plate) and train a binary classifier for plate/non-plate classification problem for a pre-defined image size.
In test time we need to evaluate (apply the trained classifier)  every position in an image in different scales to find the true position and size of the plate.

There are more efficient methods for doing the detection. One is creating an AI Deep Learning Model for this task.
Note: the size of the database is too small for training large architectures from scratch. Thus, I'm using Transfer Learnig in this project.


Main Steps:

1. Data Preparation and Pre-Processing: depends on the chosen model

2. Training and evaluation of the build model.

3. Presenting the results and hypothesis of the AI model on training and test datasets.

In this project I've used Transfer Learning - VGG16 Backbone and added a few trainable layers on top.


**Important Note:
**In the license_plate_detection.py script, user needs to call the (_pre_processing) function at least once when running the code.
  The data_loading() function assumes that some files have been created and saved, and will try to load them.
  The problem is that one specific file - Dataset/images.npy wasn't added to Git due to large size (> 100MB).
  So, run it once, for the mentioned file to be created and after, one can put it in commit and not use it!
