#%%

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
from keras.losses import MeanAbsoluteError
from keras.layers import Reshape, Flatten, LeakyReLU, Activation, Dense, BatchNormalization,Conv2D,Conv2DTranspose,AveragePooling2D,MaxPooling2D
from keras.regularizers import L1L2
from keras.layers.convolutional import UpSampling2D
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD,Adam,RMSprop
import tensorflow as tf

from keras.layers import Dropout

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

def define_generator(latent_dim):
    model = Sequential()
    nch = 256
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)
    h = 5
    n1=0.1
    batch_norm=tf.keras.layers.experimental.SyncBatchNormalization()
    model.add(Dense(nch * 4 * 4, input_dim=100, kernel_regularizer=reg()))
    model.add(tf.keras.layers.experimental.SyncBatchNormalization())
    model.add(Reshape(dim_ordering_shape((nch, 4, 4))))
    model.add(Conv2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(tf.keras.layers.experimental.SyncBatchNormalization())
    model.add(LeakyReLU(n1))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(tf.keras.layers.experimental.SyncBatchNormalization())
    model.add(LeakyReLU(n1))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(int(nch / 4), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(tf.keras.layers.experimental.SyncBatchNormalization())
    model.add(LeakyReLU(n1))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(int(nch / 8), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(tf.keras.layers.experimental.SyncBatchNormalization())
    model.add(LeakyReLU(n1))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(int(nch / 8), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(tf.keras.layers.experimental.SyncBatchNormalization())
    model.add(LeakyReLU(n1))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (h, h), padding='same', kernel_regularizer=reg()))
    model.add(Activation('tanh'))
    return model

def define_gan(g_model, d_model):

    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)

    return model

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples).reshape(n_samples,latent_dim)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y


def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y

def generate_real_fake_famples(dataset,g_model, latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples).reshape(n_samples, latent_dim)
    ix = np.random.randint(0, dataset.shape[0], n_samples)

    X1 = g_model.predict(x_input)
    y1 = np.zeros((n_samples, 1))

    X2 = dataset[ix]
    y2 = np.ones((n_samples, 1))

    X=np.append(X1,X2)
    Y=np.append(y1,y2)

    X_new=np.zeros((n_samples,128,128,3))
    Y_new=np.zeros((n_samples,1))
    for i in range(n_samples):
        j=np.random.randint(n_samples)
        X_new[i]=X[j]
        Y_new[i]=Y[j]

    return X_new, Y_new

def train(g_model, d_model, gan_model, dataset, latent_dim, n_batch=16):
    loss_arr=[]
    half_batch = int(n_batch / 2)
    epochs = 10000
    batch_size=int(dataset.shape[0] / n_batch)

   # g_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)
  #  d_optim = SGD(lr=0.005, momentum=0.9, nesterov=True)

    #g_optim = Adam(lr=0.01, beta_1=0.3)
    #d_optim = Adam(lr=0.01, beta_1=0.3)



    d_optim=RMSprop()
    g_optim=RMSprop()

    g_loss=MeanAbsoluteError()


    gan_model.compile(loss='binary_crossentropy', optimizer=g_optim)

    d_model.compile(loss='binary_crossentropy', optimizer=d_optim, metrics=['categorical_accuracy'])

    for i in range(epochs):
        for j in range(batch_size):

            X_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)

            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

            X_fake, y_fake = generate_real_fake_famples(dataset,g_model, latent_dim, half_batch)
            d_loss3, _ = d_model.train_on_batch(X_fake, y_fake)

            X_gan = randn(latent_dim * n_batch).reshape(n_batch,latent_dim)
            y_gan = np.ones((n_batch, 1))
            gan_loss = gan_model.train_on_batch(X_gan, y_gan)
        if i %2 ==0:
               print('epochs:%d, %d in %d, real loss=%.3f, fake loss=%.3f,fake+real loss=%.3f ,gan loss=%.3f' %
                   (i , j , batch_size, d_loss1, d_loss2,d_loss3, gan_loss))
               g_model.save_weights('weights/generator.hdf5', True)
               d_model.save_weights('weights/discriminator.hdf5', True)
               loss_arr.append((i , j , batch_size, d_loss1, d_loss2,d_loss3, gan_loss))
               np.save('weights/losses.npy', np.array(loss_arr))


def generate(latent_dim,n_batch):
    generator = define_generator(latent_dim)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator1')

    noise = randn(latent_dim * n_batch).reshape(n_batch, latent_dim)
    # generate images
    X = generator.predict(noise)

    X = X*127.5+127.5
    image = X.reshape((128,128,3))
    Image.fromarray(image.astype(np.uint8)).save("jupit.png")

def process_data(pokemon_dir,shape):
    N = len(os.listdir(pokemon_dir))

    images = np.zeros((N, shape[0],shape[1],shape[2]))
    for i, each in enumerate(os.listdir(pokemon_dir)):
        images[i] = io.imread(pokemon_dir + '/' + each)

    images = images.astype('float32')

    num_images = len(images)

    return images, num_images


if __name__ == '__main__':
    #
    # default generator and discriminator image shape=128,128,3
    #
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    BATCH_SIZE_PER_REPLICA = 8
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    latent_dim = 100
    image_shape = 128, 128, 3
    data_dir = '../resizedPokemons'

    (xtrain, ytrain) = process_data(data_dir, image_shape)
    xtrain = xtrain.astype('float32')
    xtrain = (xtrain - 127.5) / 127.5

    with strategy.scope():
        d_model = define_discriminator()
        d_model.summary()
        g_model = define_generator(latent_dim)
        g_model.summary()
        gan_model = define_gan(g_model, d_model)

    train(g_model, d_model, gan_model, xtrain, latent_dim, BATCH_SIZE)

    # generate(latent_dim,n_batch=1)

