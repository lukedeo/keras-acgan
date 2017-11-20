"""
file: ACGAN-CIFAR10.py
author: Luke de Oliveira (lukedeo@vaitech.io)
contributor: KnightTuYa (398225157@qq.com)
Consult https://github.com/lukedeo/keras-acgan for MNIST version!
Consult https://github.com/soumith/ganhacks for GAN trick!
"""
from __future__ import print_function
import os
from collections import defaultdict
import pickle as pickle
from PIL import Image

from six.moves import range

import keras.backend as K
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, Activation, Conv2DTranspose, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils.generic_utils import Progbar
import matplotlib.pyplot as plt
from keras.layers.noise import GaussianNoise
import numpy as np

np.random.seed(1337)
class_num = 10
K.set_image_dim_ordering('th')
path = "output"
save = 950

def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    cnn = Sequential()
    cnn.add(Dense(384, input_dim=(latent_size), activation='relu'))
    cnn.add(Dense(192 * 4 * 4, activation='relu'))
    cnn.add(Reshape((192, 4, 4)))

    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(192, 5, border_mode='same', activation='relu', init='glorot_normal'))
    cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))

    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(96, 5, border_mode='same',
                          activation='relu', init='glorot_normal'))
    cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))


    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(3, 5, border_mode='same',
                          activation='tanh', init='glorot_normal'))
    # cnn.add(Dropout(0.5))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(10, latent_size,
                              init='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return Model(input=[latent, image_class], output=fake_image)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    cnn.add(GaussianNoise(0.005, input_shape=(3, 32, 32))) #Add this layer to prevent D from overfitting! 
    cnn.add(Convolution2D(16, 3, 3, border_mode='same', subsample=(2, 2), init='glorot_normal'))
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Convolution2D(32, 3, 3, border_mode='same', init='glorot_normal', subsample=(1, 1)))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Convolution2D(64, 3, 3, border_mode='same', init='glorot_normal', subsample=(2, 2)))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Convolution2D(128, 3, 3, border_mode='same', init='glorot_normal', subsample=(1, 1)))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Convolution2D(256, 3, 3, border_mode='same', init='glorot_normal', subsample=(2, 2)))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Convolution2D(512, 3, 3, border_mode='same', init='glorot_normal', subsample=(1, 1)))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Flatten())

    image = Input(shape=(3, 32, 32))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(class_num, activation='softmax', name='auxiliary')(features)

    return Model(input=[image], output=[fake, aux])

if __name__ == '__main__':

    # batch and latent size taken from the paper
    nb_epochs = 500
    batch_size = 100
    latent_size = 100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator, Choose Adam as optimizer according to GANHACK
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr= adam_lr, beta_1=adam_beta_1, decay=1e-7),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # build the generator, Choose RMSprop as optimizer according to GANHACK
    generator = build_generator(latent_size)
    generator.compile(optimizer=RMSprop(lr=adam_lr, decay=1e-7),
                      loss='binary_crossentropy')

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model(input=[latent, image_class], output=[fake, aux])

    combined.compile(
        optimizer=RMSprop(lr= adam_lr, decay=1e-7),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

   
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)
    generator.load_weights('params_generator_epoch_{0:03d}.hdf5'.format(save-1))
    discriminator.load_weights('params_discriminator_epoch_{0:03d}.hdf5'.format(save-1))
    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            progress_bar.update(index)
            # generate a new batch of noise
            noise = np.random.normal(0, 0.5, (batch_size, latent_size))

            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, class_num, batch_size)

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)
			
			# According to GANHACK, We training our ACGAN-CIFAR10 in Real->D, Fake->D, 
			# Noise->G, rather than traditional method: [Real, Fake]->D, Noise->G, actully, 
			# it really make sense!
			# Label Soomthing
            X_real = image_batch
            y_real = np.random.uniform(0.7, 1.2, size=(batch_size, ))
            aux_y1 = label_batch.reshape(-1, )

            epoch_disc_loss.append(discriminator.train_on_batch(X_real, [y_real, aux_y1]))
			# Label Soomthing
            X_fake = generated_images
            y_fake = np.random.uniform(0.0, 0.3, size=(batch_size, ))
            aux_y2 = sampled_labels
            
            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(X_fake, [y_fake, aux_y2]))

            # make new noise. we generate Guassian Noise rather than Uniforn Noise according to GANHACK
            noise = np.random.normal(0, 0.5, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, class_num, 2 * batch_size)

            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        # evaluate the testing loss here

        # generate a new batch of noise
        noise = np.random.normal(0, 0.5, (nb_test, latent_size))

        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, class_num, nb_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        X = np.concatenate((X_test, generated_images))
        y = np.array([np.random.uniform(0.7, 1.2)] * nb_test + [np.random.uniform(0.0, 0.3)] * nb_test)
        aux_y = np.concatenate((y_test.reshape(-1, ), sampled_labels), axis=0)


        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        noise = np.random.normal(0, 0.5, (2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, class_num, 2 * nb_test)

        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights(
            'params_generator_epoch_{0:03d}.hdf5'.format(epoch+save), True)
        discriminator.save_weights(
            'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch+save), True)

        # generate some digits to display
        noise = np.tile(np.random.normal(0, 0.5,  (1, latent_size)), (100, 1))
        sampled_labels = np.array([
            [i] * 10 for i in range(10)
        ]).reshape(-1, 1)
        generated_images = generator.predict([noise, sampled_labels]).transpose(0, 2, 3, 1)
        #print(generated_images.shape)
        generated_images = np.asarray((generated_images*127.5+127.5).astype(np.uint8))
        '''
        generated = generated_images.reshape((32, 32, 3))
        generated = generated * 127.5 + 127.5
        generated = np.asarray(generated, dtype=np.uint8)
        img = Image.fromarray(generated)
        img.save("images/generated"+str(epoch)+".png")
        #for r in np.split(generated_images, 10):
         #   print (r.reshape(-1, 32, 32, 3).shape)
        # get a batch to display
        '''
        #print(generated_images.shape)
        '''
        def generator_sampler():
            xpred = dim_ordering_unfix(generated_images=
                generator.predict([noise, sampled_labels])).transpose((0, 2, 3, 1))
            return xpred.reshape((10, 10) + xpred.shape[1:])
        
        generator_cb = ImageGridCallback(os.path.join(path, "epoch-{:03d}.png"), generator_sampler, cmap=None)
        '''

        '''
        a = generated_images[1, :, :, :].reshape(3, 128, 128)
        a = a.transpose(1, 2, 0)
        a = a * 127.5 + 127.5
        img = np.uint8(a)
        plt.figure()
        plt.imshow(img)
        plt.show()
        for r in np.split(generated_images, 10):
            #print (r.shape)
        '''

        def vis_square(data, padsize=1, padval=0):

            # force the number of filters to be square
            n = int(np.ceil(np.sqrt(data.shape[0])))
            padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
            data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

            # tile the filters into an image
            data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
            data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
            return data
        img = vis_square(generated_images)
        Image.fromarray(img).save(
            'images/plot_epoch_{0:03d}_generated.png'.format(epoch+save))

    pickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))
