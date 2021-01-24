

import numpy as np
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from skimage import io
import tensorflow as tf
import math
import os
from PIL import Image
from image_utils import dim_ordering_fix, dim_ordering_unfix, dim_ordering_shape
from keras.models import Sequential
from keras.layers import Reshape, Flatten, LeakyReLU, Activation, Dense, BatchNormalization,Conv2D,Conv2DTranspose,AveragePooling2D,MaxPooling2D
from keras.regularizers import L1L2
from keras.layers.convolutional import UpSampling2D
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD,Adam,RMSprop


from keras.layers import Dropout



# дискриминатор
def define_discriminator():
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)
    model = Sequential()

    model.add(GaussianNoise(0.1, input_shape=( 128,128,3)))

    model.add(Conv2D(64, (5, 5), padding='same', kernel_regularizer=reg()))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, (5, 5), padding='same', kernel_regularizer=reg()))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, (5, 5), padding='same', kernel_regularizer=reg()))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(1, (5, 5), padding='same', kernel_regularizer=reg()))
    model.add(AveragePooling2D(pool_size=(4, 4), padding='valid'))
    model.add(Activation('tanh'))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))


    return model


#генератор
def define_generator(latent_dim):
    model = Sequential()
    nch = 256
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)
    h = 5
    n1=0.1
    model.add(Dense(nch * 4 * 4, input_dim=100, kernel_regularizer=reg()))
    model.add(BatchNormalization())
    model.add(Reshape(dim_ordering_shape((nch, 4, 4))))
    model.add(Conv2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(n1))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(n1))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(int(nch / 4), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(n1))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(int(nch / 8), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(n1))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(int(nch / 8), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(n1))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (h, h), padding='same', kernel_regularizer=reg()))
    model.add(Activation('tanh'))
    return model

#генератор+дискриминатор
def define_gan(g_model, d_model):

    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)

    return model

def generate(latent_dim,n_batch):
    generator = define_generator(latent_dim)
    generator.load_weights('weights/generator.hdf5')

    noise = randn(latent_dim * n_batch).reshape(n_batch, latent_dim)
    # generate images
    X = generator.predict(noise)

    X = X*127.5+127.5
    image = X.reshape((128,128,3))
    Image.fromarray(image.astype(np.uint8)).save("pokemon.jpg")

# для датасета с крупными картиночками
def process_data():
    size1, size2, channel = 128, 128, 3

    pokemon_dir = 'resized'

    N = len(os.listdir(pokemon_dir))

    images = np.zeros((N, size1, size2, channel))
    for i, each in enumerate(os.listdir(pokemon_dir)):
        images[i] = io.imread(pokemon_dir + '/' + each)

    images = images.astype('float32')

    num_images = len(images)

    return images, num_images


if __name__=='__main__':

  latent_dim = 100

  d_model = define_discriminator()
  d_model.summary()
  g_model = define_generator(latent_dim)
  g_model.summary()
  gan_model = define_gan(g_model, d_model)


  (xtrain, ytrain) = process_data()

  xtrain=xtrain.astype('float32')

  xtrain=(xtrain-127.5)/127.5

  #train(g_model, d_model, gan_model, xtrain, latent_dim,n_batch=16)

  generate(latent_dim,n_batch=1)
