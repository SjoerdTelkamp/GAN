# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:04:44 2019

@author: s138069
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:34:04 2019

@author: admin
"""

import matplotlib.pyplot as plt
import os, time  
import numpy as np 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tqdm import tqdm
import tensorflow as tf

dir_data      = "Data\img_align_celeba"
Ntrain        = 10000 
Ntest         = 100
nm_imgs       = np.sort(os.listdir(dir_data))
## name of the jpg files for training set
nm_imgs_train = nm_imgs[:Ntrain]
## name of the jpg files for the testing data
nm_imgs_test  = nm_imgs[Ntrain:Ntrain + Ntest]
#set target images size 
img_shape     = (28, 28, 3)

#read images and shrink to target size
def get_npdata(nm_imgs_train):
    X_train = []
    for i, myid in enumerate(nm_imgs_train):
        image = load_img(dir_data + "/" + myid,
                         target_size=img_shape[:2])
        image = img_to_array(image)/255.0
        image = np.ndarray.flatten(image)
        X_train.append(image)
    X_train = np.array(X_train)
    
    return(X_train)

X_train = get_npdata(nm_imgs_train)
print("X_train.shape = {}".format(X_train.shape))

X_test  = get_npdata(nm_imgs_test)
print("X_test.shape = {}".format(X_test.shape))

#2 layer simple GAN
#------------------------------------------


optimizer = Adam(0.0002,0.5)

#noise imput for generator
def get_noise(nsample=1, noise_dim=100):
    noise = np.random.normal(0, 1, (nsample,noise_dim))
    return(noise)

noise_shape = (100,)

#Generator 3 layer neural net
#------------------------------------------------------------
def create_generator():
    generator=Sequential()
    generator.add(Dense(units=768,input_dim=100))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=1536))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=3072))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=2352, activation='tanh'))
 
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator
g=create_generator()
g.summary()

#Discriminator 3 layer neural net
#-------------------------------------------
def create_discriminator():
    discriminator=Sequential()
    discriminator.add(Dense(units=3072,input_dim=2352))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    
    discriminator.add(Dense(units=1536))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(units=768))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator
d =create_discriminator()
d.summary()

#now the two combine in a GAN
def create_GAN(discriminator, generator):
    discriminator.trainable=False
    GAN_in = Input(shape=(100,))
    x = generator(GAN_in)
    GAN_out= discriminator(x)
    GAN= Model(inputs=GAN_in, outputs=GAN_out)
    GAN.compile(loss='binary_crossentropy', optimizer='adam')
    return GAN
GAN = create_GAN(d,g)
GAN.summary()

#plot images 
def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100,28,28,3)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('GAN_img %d.png' %epoch)


#Training the GAN

def training(epochs=1, batch_size=128):
    (X_train)= get_npdata(nm_imgs_train)

    generator= create_generator()
    discriminator= create_discriminator()
    GAN = create_GAN(discriminator, generator)
    for e in range(1, epochs+1):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
            noise = np.random.normal(0,1, [batch_size, 100])
            image_batch = X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
            generated_images = generator.predict(noise)
            X_batch= np.concatenate([image_batch, generated_images])
            #creating y labels 
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=0.9
            #check if models runs on data and noise 
            discriminator.trainable=True
            discriminator.train_on_batch(X_batch, y_dis)
            noise= np.random.normal(0,1, [batch_size, 100])
            y_gen = np.ones(batch_size)
            discriminator.trainable=False
            GAN.train_on_batch(noise, y_gen)
            if e == 1 or e % 20 == 0:
                plot_generated_images(e, generator)   
training(400,128)
  
#
##Tensorflow optie
#Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
#initializer = tf.contrib.layers.xavier_initializer()
#Gw1 = tf.Variable(initializer[100, 128], name='Gw1')
#Gb1 = tf.Variable(tf.zeros(shape=[128]), name='Gb1')
#    
#
#Gw2 = tf.Variable(initializer[128, 784], name='Gw2')
#G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')
#
#theta_G = [G_W1, G_W2, G_b1, G_b2]
#
#
#def generator(z):
#    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
#    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
#    G_prob = tf.nn.sigmoid(G_log_prob)
#
#    return G_prob
