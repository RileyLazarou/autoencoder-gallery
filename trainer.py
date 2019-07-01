import tensorflow.keras as K
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import models
from PIL import Image
import matplotlib.pyplot as plt

def train_test_split(data, test_p=0.2):
	full_data = np.copy(data[:])
	indices = np.arange(full_data.shape[0])
	np.random.shuffle(indices)
	test_data = full_data[indices[:int(len(full_data)*test_p)]]
	train_data = full_data[indices[int(len(full_data)*test_p):]]
	return train_data, test_data

def load_data(directory):
	# Open dir and grab all PNG, assert all have same dimensions
	# Convert to 4D numpy array
	images = []
	for root, dirs, files in os.walk(directory):
		for index, f in enumerate(files):
			if index % 100 == 0:
				print(index, end='\r')
			if f.split(".")[-1] == "png":
				im = np.asarray(Image.open(directory + "/" + f))
				if len(im.shape) == 2:
					#convert greyscale image to single channel
					im = im.reshape(*im.shape, 1)
				if np.any(im > 1.0):
					#convert [0,255] to [0.0, 1.0]
					im = im / 255.0
				images.append(im)
	images = np.array(images)
	if images.shape[-1] == 4 and np.all(images[..., 3] == 1.0):
		# remove uninformative alpha
		images = images[..., :3]
	if np.all(images[..., 0] == images[..., 1]) and np.all(images[..., 1] == images[..., 2]):
		#image is rgb greyscale; convert to single channel
		images = images[..., :1]
	if len(images.shape) != 4:
		raise ValueError("Images must all have the same width and height.")
	return images

def save_result(filename, images, reconstructed, num=10, invert=False):
	if num % 2 != 0:
		raise ValueError("num must be an even number.")

	channels = images.shape[-1]
	if channels == 1:
		mode = 'L'
	elif channels == 3:
		mode = 'RGB'
	elif channels == 4:
		mode = 'RGBA'

	if invert:
		images = 1 - images
		reconstructed = 1 - reconstructed


	# create grid of images with alternating columns: original, reconstructed ...
	length = images.shape[1]

	imgs_per_output = int((num*num)/2)
	output_image = np.zeros((length*num, length*num, images.shape[3]))
	print(output_image.shape)
	for i in range(imgs_per_output):
		selected = np.random.randint(images.shape[0])
		row = i % num
		col = (i // num)*2
		output_image[row*length: (row+1)*length , col*length: (col+1)*length] = images[selected, :, :, :]
		output_image[row*length: (row+1)*length , (col+1)*length: (col+2)*length] = reconstructed[selected, :, :, :]

	output_image = np.round(output_image * 255)
	output_image = output_image.astype(np.uint8)
	if mode == "L":
		output_image = output_image.reshape((length*num, length*num))
	else:
		output_image = output_image.reshape((length*num, length*num, channels))
	output_image = Image.fromarray(output_image, mode=mode)
	output_image.save(filename, mode=mode)

def train_model(model, data):
	model.fit(data, data, epochs=1000, validation_split=0.2, callbacks=[EarlyStopping(patience=5)], batch_size=128)

def test_model():
	pass

if __name__ == "__main__":
	
	data = load_data(os.getcwd() + "/data/chinese_serif_32")

	# dimensions, channels, latent_dim
	model = models.build_conv_ae(data.shape[1], data.shape[-1], 32)

	train, test = train_test_split(data)

	train_model(model, train)
	reconstructed = model.predict(test)

	save_result("output_char_comparison.png", test, reconstructed, invert=False)