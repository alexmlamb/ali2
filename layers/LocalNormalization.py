import theano
import theano.tensor as T
import numpy as np
import numpy.random as random
from theano.sandbox.cuda import dnn

'''


var(x) = E[X^2] - E[X]^2

'''
k = 3


#batch, filter, X, Y

def local_mean(x,n_in):

    W_mean = theano.shared((np.ones(shape = (1,n_in,k,k)) / (k*k*500)).astype('float32'))

    #out, in, kernel, kernel

    padsize = 'half'

    conv_out = dnn.dnn_conv(img=x,kerns=W_mean,subsample=(1,1),border_mode=padsize, precision = 'float32')   

    conv_out = T.set_subtensor(conv_out[:,:,:2,:], conv_out[:,:,2:4,:])
    conv_out = T.set_subtensor(conv_out[:,:,-2:,:], conv_out[:,:,-4:-2,:])

    conv_out = T.set_subtensor(conv_out[:,:,:,:2], conv_out[:,:,:,2:4])
    conv_out = T.set_subtensor(conv_out[:,:,:,-2:], conv_out[:,:,:,-4:-2])

    return T.addbroadcast(conv_out, 1)

def local_stdv(x, n_in):
    W_mean = theano.shared((np.ones(shape = (1,n_in,k,k)) / (k*k*500)).astype('float32'))

    return T.addbroadcast(local_mean(x**2, n_in) - local_mean(x, n_in)**2, 1)

if __name__ == "__main__":

    x = random.normal(5.0, 1.0, size = (10,500,30,30)).astype('float32')

    xu = T.tensor4()
    c = local_mean(xu, 500)
    stdv = local_stdv(xu, 500)

    out = (xu - c) / stdv

    f = theano.function([xu], [c, stdv, out])

    r = f(x)

    print r[2].shape

    print r[2]


