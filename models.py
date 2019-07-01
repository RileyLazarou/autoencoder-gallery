import tensorflow.keras as K
import numpy as np
import os
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


def build_conv_ae(dim, channels, latent_dim):
	if dim < 16:
		raise ValueError("Image dimensions must be at least 16x16.")
	if channels < 1:
		raise ValueError("Channels must be a positive integer.")
	if latent_dim < 1:
		raise ValueError("Latent dimension must be a positive integer.")


	# input layer
	input_layer = layers.Input((dim, dim, channels))
	X = input_layer

	# conv layers
	half_dim = dim
	counter = 0
	# Encoding
	while half_dim >= 8:
		# make layers

		# Conv2D(num_channels, window size, stride)
		X = layers.Conv2D(16*2**(counter), 3, 1, padding='same')(X)
		X = layers.Activation('relu')(X)
		X = layers.Conv2D(16*2**(counter + 1), 3, 2, padding='same')(X)
		X = layers.Activation('relu')(X)
		counter += 1
		half_dim = np.ceil(half_dim / 2)

	# End of encoding
	X = layers.Flatten()(X)
	latent_space = layers.Dense(latent_dim)(X)
	X = layers.Dense(half_dim * half_dim * 16*2**(counter))(latent_space)
	X = layers.Reshape((half_dim, half_dim, 16*2**(counter)))(X)

	for i in range(counter):
		X = layers.Conv2DTranspose(16*2**(counter-i), 4, 2, padding='same')(X)
		X = layers.Activation('relu')(X)
		X = layers.Conv2DTranspose(16*2**(counter-i-1), 3, 1, padding='same')(X)
		X = layers.Activation('relu')(X)

	X = layers.Conv2D(channels, 3, 1, padding='same')(X)
	X = layers.Activation('sigmoid')(X)

	# crop layer
	reconstructed_dim = half_dim * 2 ** counter
	left_diff = int((reconstructed_dim-dim) / 2)
	right_diff = (reconstructed_dim-dim) - left_diff
	output_layer = layers.Cropping2D(((left_diff, right_diff), (left_diff, right_diff)))(X)
	
	# output layer
	model = models.Model(input_layer, output_layer)
	model.compile('adam', loss='MAE')

	return model


def build_vae():
	pass

def build_beta_vae():
	pass



if __name__ == "__main__":
 	pass