from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import pickle
import sys

import numpy as np

class textGAN():
    def __init__(self):
        self.txt_rows = 51 
        self.txt_cols = 1
        self.channels = 1

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated txts
        z = Input(shape=(51,1))
        txt = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(txt)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (51,1)
        
        model = Sequential()

        model.add(LSTM(256, input_shape=noise_shape))
        model.add(Dropout(0.2))
        model.add(Dense(noise_shape[0], activation='tanh'))

        model.summary()

        noise = Input(shape=noise_shape)
        txt = model(noise)

        return Model(noise, txt)

    def build_discriminator(self):

        txt_shape = (self.txt_rows, self.txt_cols, self.channels)
        
        model = Sequential()

        model.add(Flatten(input_shape=txt_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        txt = Input(shape=txt_shape)
        validity = model(txt)

        return Model(txt, validity)

    def train(self, epochs, batch_size, save_interval):
        #load data
        x = pickle.load(open("cerpen.p","rb"))
        x_train = x[0]
        int2word = x[5]
        # Rescale -1 to 1
        X_train = (x_train.astype(np.float32) - 2005.5)/2005.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            txts = X_train[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, 51,1))
            gen_txts = self.generator.predict(noise)
            gen_txts = np.expand_dims(gen_txts, axis=3)
            gen_txts = np.expand_dims(gen_txts, axis=4)
            #print(X_train[10])
            #print(gen_txts[0])
            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(txts, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_txts, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 51,1))

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch%save_interval == 0 :  
                self.save_txts(epoch, int2word)
            elif epoch==epochs:
                self.save_txts(epoch,int2word)

    def save_txts(self, epoch, int2word):
        r= 10
        noise = np.random.normal(0, 1, (r , 51,1))
        gen_txts = self.generator.predict(noise)
        gen_txts = np.expand_dims(gen_txts, axis=3)
        
        # Rescale text 0 - 4011
        gen_txts = 2005.5 * gen_txts + 2005.5
        gen_txts = np.round(gen_txts)
        gen_txts = gen_txts.astype(int)
        #print(np.amax(gen_txts[0]))
        #index = gen_txts[0].ravel()
        #print(index)
        text_file = open("text/text epoch %d.txt" %epoch, "w")
        texts = [None] * r
        for i in range(r):
            texts[i] = []
            index = gen_txts[i].ravel()
            for j in range(51):
                texts[i].append(int2word[index[j]])
        strings = [None] * r
        for i in range(r):
            strings[i] = ' '.join(texts[i])

        sentence = '\n'.join(strings)
        text_file.write(sentence)
        text_file.close()


if __name__ == '__main__':
    textgan = textGAN()
    textgan.train(epochs=2000, batch_size=1024, save_interval=50)