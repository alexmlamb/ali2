from fuel.datasets.youtube_audio import YouTubeAudio
import random

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

import scipy

import gzip

import cPickle as pickle

from plot import plot_image_grid

import scipy.io

import h5py

from matplotlib import cm, pyplot

class Data:

    def __init__(self, mb_size):
        self.mb_size = mb_size

        file_loc_train = "/u/lambalex/data/stl/train.mat"
        file_loc_test = "/u/lambalex/data/stl/test.mat"

        datatrain = scipy.io.loadmat(file_loc_train)
        datatest = scipy.io.loadmat(file_loc_test)


        file_loc_unlabeled = "/u/lambalex/data/stl/unlabeled.mat"

        UX = h5py.File(file_loc_unlabeled)['X'].value.T

        #np.transpose(np.swapaxes(train_X,1,0).reshape((100000, 3, 96, 96)), (0,3,2,1))

        #100k
        self.unlabeled_X = UX.reshape((100000, 3, 96, 96)).transpose(0,1,3,2)
        self.unlabeled_X = (self.unlabeled_X[:,:,:96,:96] / 300.0).astype('float32')

        self.train_X = datatrain['X'].reshape((5000, 3, 96, 96)).transpose(0,1,3,2)
        self.train_X = (self.train_X[:,:,:96,:96] / 300.0).astype('float32')
        self.train_Y = datatrain['y'] - 1

        self.test_X = datatest['X'].reshape((8000, 3, 96, 96)).transpose(0,1,3,2)
        self.test_X = (self.test_X[:,:,:96,:96] / 300.0).astype('float32')
        self.test_Y = datatest['y'] - 1

    '''
        Pick a random location, get sequence of seq_length
    '''
    def getBatch(self, segment):

        if segment == "train":
            x_use, y_use = self.train_X, self.train_Y
        elif segment == "test":
            x_use, y_use = self.test_X, self.test_Y
        elif segment == "unlabeled":
            x_use, y_use = self.unlabeled_X, None
        else:
            raise Exception('segment not found')

        startingPoint = random.randint(0, x_use.shape[0] - self.mb_size - 1)

        out = {"x" : x_use[startingPoint : startingPoint + self.mb_size]}

        if segment == "unlabeled":
            out['y'] = None
        else:
            out['y'] = y_use[startingPoint : startingPoint + self.mb_size]

        return out

    def saveExample(self, x_gen, name):

        assert x_gen.ndim == 4

        x_gen = x_gen * 300.0

        x_gen = np.clip(x_gen, 1.0, 250.0)

        print x_gen.min(), x_gen.max()

        imgLoc = "plots/" + name + ".png"

        #scipy.misc.imsave(imgLoc, x_gen[0])

        x_gen = x_gen.astype('uint8')

        images = []

        for i in range(0, x_gen.shape[0]):
            images.append(x_gen[i])

        plot_image_grid(images[:3*6], 3, 6, imgLoc)

if __name__ == "__main__":
    d = Data(mb_size = 128)

    r = d.getBatch("unlabeled")

    x = r['x']

    #print r['y'].shape

    print x.shape

    print x.dtype

    d.saveExample(x, name = 'derp')

    d.saveExample((x * 1.3), name = 'derp2')

