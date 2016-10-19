import time
import numpy as np
import lasagne
import theano
import theano.tensor as T
from cifar import load_cifar10
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm


def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            padded = np.pad(inputs[excerpt], ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0, high=8, size=(batchsize, 2))
            for r in range(batchsize):
                random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32),
                                                    crops[r, 1]:(crops[r, 1] + 32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]


def build_cnn(input_X, input_shape):
    input_layer = lasagne.layers.InputLayer(shape=input_shape, input_var=input_X)

    conv_layer = batch_norm(lasagne.layers.Conv2DLayer(input_layer,
                                                       num_filters=10,
                                                       filter_size=(3, 3),
                                                       stride=(3, 3),
                                                       nonlinearity=rectify,
                                                       pad='same',
                                                       W=lasagne.init.HeNormal(gain='relu'),
                                                       flip_filters=False))

    dense_1 = lasagne.layers.DenseLayer(conv_layer,
                                        num_units=100,
                                        nonlinearity=lasagne.nonlinearities.sigmoid,
                                        W=lasagne.init.HeNormal(gain='relu'),
                                        name="hidden_dense_layer")

    do_layer = lasagne.layers.DropoutLayer(dense_1, p=0.1)
    #fully connected output layer that takes dense_1 as input and has 10 neurons (1 for each digit)
    #We use softmax nonlinearity to make probabilities add up to 1

    dense_output = lasagne.layers.DenseLayer(do_layer,
                                             num_units=10,
                                             nonlinearity=softmax,
                                             name='output')
    return dense_output


input_X = T.tensor4("X")
input_shape = [None, 3, 32, 32]
target_y = T.vector("target Y integer", dtype='int32')


dense_output = build_cnn(input_X, input_shape)

#network prediction (theano-transformation)
y_predicted = lasagne.layers.get_output(dense_output)

all_weights = lasagne.layers.get_all_params(dense_output, trainable=True)

#Mean categorical crossentropy as a loss function - similar to logistic loss but for multiclass targets
loss = lasagne.objectives.categorical_crossentropy(y_predicted, target_y).mean()

#prediction accuracy (WITH dropout)
accuracy = lasagne.objectives.categorical_accuracy(y_predicted, target_y).mean()

#This function computes gradient AND composes weight updates just like you did earlier
updates_sgd = lasagne.updates.adam(loss, all_weights)

#function that computes loss and updates weights
train_fun = theano.function([input_X, target_y], [loss, accuracy], updates=updates_sgd)

#deterministic prediciton (without dropout)
y_predicted_det = lasagne.layers.get_output(dense_output, deterministic=True)

#prediction accuracy (without dropout)
accuracy_det = lasagne.objectives.categorical_accuracy(y_predicted_det, target_y).mean()

#function that just computes accuracy without dropout/noize -- for evaluation purposes
accuracy_fun = theano.function([input_X, target_y], accuracy_det)

num_epochs = 20  # amount of passes through the data

batch_size = 500  # number of samples processed at each function call

X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10("cifar_data")
class_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])


for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_acc = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, batch_size):
        inputs, targets = batch
        train_err_batch, train_acc_batch = train_fun(inputs, targets)
        train_err += train_err_batch
        train_acc += train_acc_batch
        train_batches += 1

    # And a full pass over the validation data:
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, batch_size):
        inputs, targets = batch
        val_acc += accuracy_fun(inputs, targets)
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))

    print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches))
    print("  train accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100))
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
