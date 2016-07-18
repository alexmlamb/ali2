

from simon import simon
s = simon.simon(folder = 'logs/')

s.log("derp0")

s.flush()

print "derp1"

import theano
import theano.tensor as T

print "derp2"

import data.load_stl

from layers import ConvolutionalLayer, HiddenLayer

import lasagne

import time

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
    layers = {}

    #96x96
    layers["inf_c1"] = ConvolutionalLayer.ConvPoolLayer(in_channels = 3, out_channels = 128, kernel_len = 5, stride = 2, batch_norm = True)
    #48x48
    layers["inf_c2"] = ConvolutionalLayer.ConvPoolLayer(in_channels = 128, out_channels = 256, kernel_len = 5, stride = 2, batch_norm = True)
    #24x24
    layers["inf_c3"] = ConvolutionalLayer.ConvPoolLayer(in_channels = 256, out_channels = 256, kernel_len = 5, stride = 2, batch_norm = True)
    #12x12
    layers["inf_c4"] = ConvolutionalLayer.ConvPoolLayer(in_channels = 256, out_channels = 512, kernel_len = 5, stride = 2, batch_norm = True)
    #6x6
    layers["inf_c5"] = ConvolutionalLayer.ConvPoolLayer(in_channels = 512, out_channels = 512, kernel_len = 5, stride = 2, batch_norm = True)
    #3x3

    layers['inf_h1'] = HiddenLayer.HiddenLayer(num_in = 3*3*512, num_out = 2048, batch_norm = True)

    layers["inf_disc_p"] = HiddenLayer.HiddenLayer(num_in = 2048, num_out = 1)
    layers["inf_class_p"] = HiddenLayer.HiddenLayer(num_in = 2048, num_out = 10)

    return layers

def layers2params(layers):
    params = []
    for layer in layers:
        params += layers[layer].params.values()

    return params

s.log("derp")

'''
Given x, product classification score for each class and produce a classification loss.  
'''

def make_inference_network(x, y, layers, is_labeled):

    inf_c1 = layers['inf_c1'].output(x)   

    inf_c2 = layers['inf_c2'].output(inf_c1)

    inf_c3 = layers['inf_c3'].output(inf_c2)

    inf_c4 = layers['inf_c4'].output(inf_c3)

    inf_c5 = layers['inf_c5'].output(inf_c4)

    inf_h1 = layers['inf_h1'].output(inf_c5.flatten(2))

    inf_disc_p = layers['inf_disc_p'].output(inf_h1)

    inf_class_p = T.nnet.softmax(layers['inf_class_p'].output(inf_h1))

    y = y.flatten(1)

    class_NLL = -T.mean(T.log(inf_class_p)[T.arange(y.shape[0]), y])

    class_acc = T.mean(T.eq(T.argmax(inf_class_p, axis = 1), y))

    return {'h' : 0.0, 'z' : 0, 'loss' : 0.0 * T.sum(inf_disc_p), 'class_scores' : inf_class_p, 'class_labels_true' : y, "class_NLL" : class_NLL + 0.0 * T.sum(inf_disc_p), "class_acc" : class_acc}

layers = make_layers()

x = T.tensor4()
y = T.imatrix()

inference_net_labeled = make_inference_network(x, y, layers, is_labeled = True)

params = layers2params(layers)

updates = lasagne.updates.adam(inference_net_labeled['class_NLL'], params)

print "compiling function"

train_labeled_data = theano.function([x, y], outputs = {'loss' : inference_net_labeled['loss'], 'class_scores' : inference_net_labeled['class_scores'], 'class_labels_true' : inference_net_labeled['class_labels_true'], "class_NLL" : inference_net_labeled['class_NLL'], 'class_acc' : inference_net_labeled['class_acc']}, updates = updates)

test_labeled_data = theano.function([x, y], outputs = {'loss' : inference_net_labeled['loss'], 'class_acc' : inference_net_labeled['class_acc']})

print "compiled!"

data = data.load_stl.Data(128)


for iteration in range(0,100000):

    t0 = time.time()

    training_batch = data.getBatch("train")

    x_use = training_batch['x']
    y_use = training_batch['y']

    res = train_labeled_data(x_use, y_use)

    print "iteration", iteration
    print "training loss", res['class_NLL']
    print "accuracy", res['class_acc']
    print "iteration time", time.time() - t0
    

    testing_batch = data.getBatch("test")
    x_use = testing_batch['x']
    y_use = testing_batch['y']
    res_test = test_labeled_data(x_use, y_use)
    print "test accuracy", res_test['class_acc']
