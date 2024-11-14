import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.models.load_model("../trained-models/ml_wine_reviews_FEEDFORWARD_0.keras")
