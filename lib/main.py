

from simon import simon
s = simon.simon('logs')

import theano
import theano.tensor as T

import data.load_stl

from layers import ConvolutionalLayer

'''



Types of updates: 
    -Update discriminator, labeled batch.  (classifier + increase disc acc).  
    -Update discriminator, unlabeled batch.  (increase disc acc).  
    -Update discriminator, fake batch.  (decrease disc acc).  

    -Update inference network, labeled batch.  (classifier + decrease disc acc)
    -Update inference network unlabeled batch.  (decreasse disc acc)

    -Update generator network, any batch (decrease disc acc).  

'''

def make_layers():
    pass




def make_inference_network(x, is_labeled):



    return {'h' : 0.0, 'z' : 0, 'loss' : 0}






