import tensorflow.keras as K
import numpy as np
import os
import models

def load_data(directory):
	# Open dir and grab all PNG, assert all have same dimensions
	# Convert to 4D numpy array
	pass

def train_model():
	pass

def test_model():
	pass

if __name__ == "__main__":
	
	# dimensions, channels, latent_dim
	model = models.build_conv_ae(28, 1, 8)

	model.summary()