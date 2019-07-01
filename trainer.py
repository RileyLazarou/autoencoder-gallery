import tensorflow.keras as K
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import models
import PIL

def load_data(directory):
	# Open dir and grab all PNG, assert all have same dimensions
	# Convert to 4D numpy array
	images = []
	for root, dirs, files in os.walk(directory):
		for index, f in enumerate(files):
			if index % 100 == 0:
				print(index, end='\r')
			if f.split(".")[-1] == "png":
				im = np.asarray(PIL.Image.open(directory + "/" + f))
				if len(im.shape) == 2:
					im = im.reshape(*im.shape, 1)
				im = im / 255.0
				images.append(im)
	images = np.array(images)
	if len(images.shape) != 4:
		raise ValueError("Images must all have the same width and height.")
	return images

def save_result(images):
	


def train_model(model, data):
	model.fit(data, data, epochs=1000, validation_split=0.2, callbacks=[EarlyStopping(patience=5)], batch_size=128)

def test_model():
	pass

if __name__ == "__main__":
	
	# dimensions, channels, latent_dim
	model = models.build_conv_ae(28, 1, 8)

	data = load_data(os.getcwd() + "/data/mnist")

	train_model(model, data)