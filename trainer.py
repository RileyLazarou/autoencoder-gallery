import tensorflow.keras as K
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import models
from PIL import Image
import matplotlib.pyplot as plt

def load_data(directory):
	# Open dir and grab all PNG, assert all have same dimensions
	# Convert to 4D numpy array
	images = []
	for root, dirs, files in os.walk(directory):
		for index, f in enumerate(files):
			if index % 100 == 0:
				print(index, end='\r')
				if index > 500:
					break
			if f.split(".")[-1] == "png":
				im = np.asarray(Image.open(directory + "/" + f))
				if len(im.shape) == 2:
					im = im.reshape(*im.shape, 1)
				im = im / 255.0
				images.append(im)
	images = np.array(images)
	if len(images.shape) != 4:
		raise ValueError("Images must all have the same width and height.")
	return images

def save_result(filename, images, reconstructed, num=10, invert=False):
	if num % 2 != 0:
		raise ValueError("num must be an even number.")

	if invert:
		images = 1 - images
		reconstructed = 1 - reconstructed

	# create grid of images with alternating columns: original, reconstructed ...
	length = images.shape[1]

	imgs_per_output = int((num*num)/2)
	output_image = np.zeros((length*num, length*num, images.shape[3]))
	for i in range(imgs_per_output):
		selected = np.random.randint(images.shape[0])
		row = i % num
		col = (i // num)*2
		output_image[row*length: (row+1)*length , col*length: (col+1)*length] = images[selected, :, :, :]
		output_image[row*length: (row+1)*length , (col+1)*length: (col+2)*length] = reconstructed[selected, :, :, :]

	output_image = np.round(output_image * 255)
	output_image = output_image.astype(np.uint8)
	output_image = output_image.reshape((length*num, length*num))
	output_image = Image.fromarray(output_image, mode="L")
	output_image.save(filename, mode="L")

def train_model(model, data):
	model.fit(data, data, epochs=1000, validation_split=0.2, callbacks=[EarlyStopping(patience=5)], batch_size=128)

def test_model():
	pass

if __name__ == "__main__":
	
	# dimensions, channels, latent_dim
	model = models.build_conv_ae(28, 1, 8)

	data = load_data(os.getcwd() + "/data/mnist")

	save_result("output_comparison.png", data, data, invert=True)
	#train_model(model, data)