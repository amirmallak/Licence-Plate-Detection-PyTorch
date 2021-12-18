import config

from data_exploring import exploring
from loading_data import data_loading
from AI_Model import build_AI_model


def license_plate_detection():
    WIDTH = int(config.width)  # Recommended value of 224
    HEIGHT = int(config.height)  # # Recommended value of 224
    CHANNEL = int(config.channels)

    # Loading the Dataset from Kaggle, creating a DataFrame from, and saving it into a .cvs file for later use
    # Deprecated. The Dataset on Kaggle was blocked by DataTurks!
    # URL: https://www.kaggle.com/dataturks/vehicle-number-plate-detection
    # kaggle_data_loading()

    # Pre processing the dataset images and it's corresponding .csv labels (including data filtering and cleaning)
    # pre_processing()

    # Loading the data
    dataset, mean, std = data_loading()

    # Exploring the data
    exploring(dataset, WIDTH, HEIGHT)

    # Build, Train, Validate, and Test a Neural Network AI Model
    build_AI_model(dataset, CHANNEL, WIDTH, HEIGHT)
