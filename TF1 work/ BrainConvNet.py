'''
Save and Restore a model using TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function




import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt

#import training
fname = 'C:\\Work\\PHD\\FORWARD MODEL\\Tensorflow\\trainingSetBrainForTF.mat'


with h5py.File(fname, 'r') as file:
    T1 = file.get('T1').value
    T1 = np.transpose(T1, (3, 0, 1, 2))
    T1_tf = tf.convert_to_tensor(T1, np.float64)

    T2 = file.get('T2').value
    T2 = np.transpose(T2, (3, 0, 1, 2))
    T2_tf = tf.convert_to_tensor(T2, np.float64)

    nExamples = file.get('nExamples').value

    vNorm = file.get('vNorm').value
    vNorm = np.transpose(vNorm, (1, 0))
    vNorm_tf = tf.convert_to_tensor(vNorm, np.float64)

    vPos = file.get('vPos').value
    vPos = np.transpose(vPos, (1, 0))
    vPos_tf = tf.convert_to_tensor(vPos, np.float64)

    label = file.get('label').value
    label = np.transpose(label, (1, 0))
    label_tf = tf.convert_to_tensor(label, np.float64)

    histT1 = file.get('histsT1').value
    histT1 = np.transpose(histT1, (1, 0))
    histT1_tf = tf.convert_to_tensor(histT1, np.float64)
    histT2 = file.get('histsT2').value
    histT2 = np.transpose(histT2, (1, 0))
    histT2_tf = tf.convert_to_tensor(histT2, np.float64)

    dataset = tf.data.Dataset.from_tensor_slices((T1, label))
    print("Data Loaded")


# Training Parameters
learning_rate = 0.006
num_steps = 5000
batch_size = 128
model_path = "C:\\Work\\PHD\\EEG PREPROCESS\\TensorflowModel\\model.ckpt"

# Network Parameters
num_input = 250*64 # MNIST data input (img shape: 28*28)
dropout = 0.25 # Dropout, probability to drop a unit

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        T1 = x_dict['T1']
        T2 = x_dict['T2']
        vNorm = x_dict['vNorm']
        vPos = x_dict['vPos']
        histT1 = x_dict['histT1']
        histT2 = x_dict['histT2']

        # Reshape to match MRI format [L,W,H]
        # Tensor input become 4-D: [Batchsize,L,W,H]
        xT1 = T1 #tf.reshape(T1, shape=[-1, 21, 21, 21])

        # Convolution Layer with 32 filters and a kernel size of 25,1
        conv1 = tf.layers.conv3d(xT1, 32, (3, 3, 3), activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling3d(conv1, 2, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv3d(conv1, 64, (6, 6, 6), activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling3d(conv2, 2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        flatT1 = tf.contrib.layers.flatten(conv2)


        # Reshape to match MRI format [L,W,H]
        # Tensor input become 4-D: [Batchsize,L,W,H]
        xT2 = tf.reshape(T2, shape=[-1, 21, 21, 21])

        # Convolution Layer with 32 filters and a kernel size of 25,1
        conv1 = tf.layers.conv3d(xT2, 32, (3, 3, 3), activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling3d(conv1, 2, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv3d(conv1, 64, (6, 6, 6), activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling3d(conv2, 2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        flatT2 = tf.contrib.layers.flatten(conv2)


        flatMRIs = tf.concat(flatT1,flatT2)
        # Fully connected layer (in tf contrib folder for now)
        fcMRI = tf.layers.dense(flatMRIs, 1024)
        metaVariables = tf.concat(vNorm, vPos, histT1, histT2)
        metaVariables = tf.layers.dense(metaVariables)

        fc1 = tf.concat(fcMRI, metaVariables)
        # Apply Dropout (if is_training is False, dropout is not applied)

        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, 1)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, 1, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, 1, dropout, reuse=True,
                           is_training=False)



        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.reduce_mean(tf.square(logits_train, labels)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.mean_absolute_error(labels=labels, predictions=logits_train)
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=logits_train,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'mean_absolute_error': acc_op})

    return estim_specs


# Build the Estimator
model = tf.estimator.Estimator(model_fn)
labels = label
#labels = label.flatten()

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={T1: 'T1',
       T2: 'T2',
       vNorm: 'vNorm',
       vPos: 'vPos',
       histT1: 'histT1',
       histT2: 'histT2'},
    y=labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Train the Model
i = 0
print("Step size: 1000")
while i < (num_steps/1000):
    model.train(input_fn, steps=1000)
    print("evaluated ", i*100, " examples")
    i = i+1

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': data},
    y=labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
