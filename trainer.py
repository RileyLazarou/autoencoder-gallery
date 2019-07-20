import tensorflow.keras as K
from tensorflow.keras.callbacks import EarlyStopping, Callback
import numpy as np
import os
import models
from PIL import Image
import matplotlib.pyplot as plt
from sys import argv
from time import time

VALID_AE_TYPES = ["ae", "dae", "vae", "bvae"]
OUTPUT_IMAGE_DIM = 10

class SaveImageCallback(Callback):
	"""
	"""

	def __init__(self, train_data, path, num,  frequency=1, invert=False):
		self.frequency = frequency
		self.path = path
		self.num = num
		self.invert = invert
		self.selected_images = train_data[:(num**2)//2]

	def on_epoch_end(self, epoch, logs=None):
		if epoch % self.frequency == 0:
			reconstructed = self.model.predict(self.selected_images)
			output_path = self.path.format(epoch)
			save_result(output_path, self.selected_images, reconstructed, self.num, self.invert)


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

				images.append(im)
	images = np.array(images)
	if np.any(images > 1.0):
		#convert [0,255] to [0.0, 1.0]
		images = images / 255.0
	if images.shape[-1] == 4 and np.all(images[..., 3] == 1.0):
		# remove uninformative alpha
		images = images[..., :3]
	if images.shape[1] == 3 and np.all(images[..., 0] == images[..., 1]) and np.all(images[..., 1] == images[..., 2]):
		#image is rgb greyscale; convert to single channel
		images = images[..., :1]
	if len(images.shape) != 4:
		raise ValueError("Images must all have the same width and height.")
	return images


def predict_and_save_results(epoch, model, output_path, images, num, invert):
	output_path = output_path.format(epoch)
	reconstructed = model.predict(images)
	save_result(filename, images, reconstructed, num, invert)


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
		####
		#selected = np.random.randint(images.shape[0])
		####3
		row = i % num
		col = (i // num)*2
		output_image[row*length: (row+1)*length , col*length: (col+1)*length] = images[i, :, :, :]
		output_image[row*length: (row+1)*length , (col+1)*length: (col+2)*length] = reconstructed[i, :, :, :]

	output_image = np.round(output_image * 255)
	output_image = output_image.astype(np.uint8)
	if mode == "L":
		output_image = output_image.reshape((length*num, length*num))
	else:
		output_image = output_image.reshape((length*num, length*num, channels))
	output_image = Image.fromarray(output_image, mode=mode)
	output_image.save(filename, mode=mode)

def train_model(model, data, output_path, num, save_frequency=1, save_invert=False):
	save_callback = SaveImageCallback(data, output_path, num, save_frequency, save_invert)
	early_stopping = EarlyStopping(patience=5, restore_best_weights=True, min_delta=1e-4)
	model.fit(data, data, epochs=1000, validation_split=0.2, callbacks=[early_stopping, save_callback], batch_size=128)

def test_model():
	pass

if __name__ == "__main__":
	
	# Check argv
	if len(argv) < 3:
		raise RuntimeError("Autoencoder type and data directory required.")

	ae_type = argv[1].lower()
	dataset_name = argv[2]
	
	if len(argv) > 3:
		latent_dim = int(argv[3])
	else:
		latent_dim = 32

	if len(argv) > 4:
		learning_rate = float(argv[4])
	else:
		learning_rate = 1e-3

	if len(argv) > 5:
		if argv[5] == "mae":
			loss_func = models.mae
		else:
			loss_func = argv[5]
	else:
		loss_func = models.mae

	data_dir = os.getcwd() + "/data/" + dataset_name

	if ae_type not in VALID_AE_TYPES:
		raise ValueError("Autoencoder type should be one of: {}".format(VALID_AE_TYPES))

	if not os.path.exists(data_dir):
		raise ValueError("Data directory '{}' does not exist.".format(data_dir))


	# Make directories if needed

	new_dir = os.getcwd() + "/" + ae_type + "_" + dataset_name + "_" + str(int(time()))

	os.mkdir(new_dir)
	os.mkdir(new_dir + "/images")

	print("Loading dataset ", dataset_name)
	data = load_data(data_dir)

	# dimensions, channels, latent dimensions
	model = models.build_conv_ae(data.shape[1], data.shape[-1], latent_dim, learning_rate, loss_func)

	train, test = train_test_split(data)

	train_model(model, train, "{}/images/{}_{}.{}.png".format(new_dir, ae_type, dataset_name, '{:03d}'), OUTPUT_IMAGE_DIM, save_frequency=5)
	model.save("{}/{}_{}.h5".format(new_dir, ae_type, dataset_name))

	reconstructed = model.predict(test)


	comparison_filename = "{}/{}_{}_comparison.png".format(new_dir, ae_type, dataset_name)

	save_result(comparison_filename, test, reconstructed, num=OUTPUT_IMAGE_DIM, invert=False)