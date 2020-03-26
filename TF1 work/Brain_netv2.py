from __future__ import division, print_function, absolute_import


import tensorflow as tf
import h5py
import numpy as np
#import matplotlib.pyplot as plt
import os.path

#fname = 'S:\\UQCCR-Colditz\\Signal Processing People\\Tim\\PHD\\EEG PREPROCESS\\trainingData1_2.mat'
#fname = 'C:\\Users\\Tim\\PycharmProjects\\TensorFlow-Examples\\Toydata.mat'
fname = 'C:\\Users\\Tim\\PycharmProjects\\TensorFlow-Examples\\trainingSetBrainForTF.mat'



with h5py.File(fname, 'r') as file:
    T1 = file.get('T1').value
    T2 = file.get('T2').value
    label = file.get('label').value
    if(np.any(np.isnan(T1))):
        print("warning, data contains Nan's needs to be cleaned")
    max = np.amax(T1)
    min = np.amin(T1)
    T1 = T1 - (max + min) / 2
    T1 = T1 * 2 / (max - min)

    max = np.amax(T2)
    min = np.amin(T2)
    T2 = T2 - (max + min) / 2
    T2 = T2 * 2 / (max - min)

# Training Parameters
learning_rate = 0.0003
num_steps = 10
batch_size = 1000
dropout = 0.6
num_classes = 1
n_examples = 10000

training_epochs = 100
display_epoch = 1
logs_path = 'C:\\Work\\PHD\\FORWARD MODEL\\Tensorflow\\tensorflow_logs\\example2\\'
modelPath = 'C:\\Work\\PHD\\FORWARD MODEL\\Tensorflow\\Brain_Netv2\\model.ckpt'

#reallign the input matrices
T1 = np.transpose(T1, [3, 0, 1, 2])
T1 = np.expand_dims(T1, axis=4)
T2 = np.transpose(T2, [3, 0, 1, 2])
T2 = np.expand_dims(T2, axis=4)
label = np.transpose(label)

print('T2 shape, T1 shape ', np.shape(T1), ', ', np.shape(T2))
print('label shape ', np.shape(label))

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        print('network layers')
        x = x_dict['T1']
        print(np.shape(x))
        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv3d(x, 8, 2, activation=tf.nn.relu)
        conv1 = tf.cast(conv1, tf.float32)
        conv1 = tf.layers.max_pooling3d(conv1, 2, 2)
        print(conv1.get_shape())

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv3d(conv1, 32, 2, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling3d(conv2, 2, 2)
        print(conv2.get_shape())
        # Convolution Layer with 64 filters and a kernel size of 3
        conv3 = tf.layers.conv3d(conv2, 128, 2, activation=tf.nn.relu)
        #conv3 = tf.layers.max_pooling3d(conv3, 2, 2)
        print(conv3.get_shape())
        # Convolution Layer with 64 filters and a kernel size of 3
        conv4 = tf.layers.conv3d(conv3,256, 2, activation=tf.nn.relu)
        #conv4 = tf.layers.max_pooling3d(conv4, 2, 2)
        #print(conv4.get_shape())

        # Flatten the data to a 1-D vector for the fully connected layer
        fcl = tf.reshape(conv4, [-1, np.prod(conv4.get_shape()[1:].as_list())]) #  fc1 = tf.contrib.layers.flatten(conv2) Doesnt work with a wildcard batch no. :/
        print(fcl.get_shape())
        # Output layer, class prediction
        fcl = tf.layers.dense(fcl, 15)
        fcl = tf.layers.dense(fcl, 5)
        out = tf.layers.dense(fcl, 1)

        return out

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)



        # Define loss and optimizer
    # Mean squared error
    labels = tf.cast(labels, tf.float32)
    cost = tf.reduce_mean(tf.pow(logits_train - labels, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.mean_squared_error(labels=labels, predictions=logits_test)


    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=logits_train,
        loss=cost,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs




# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, np.shape(T1)[1], np.shape(T1)[2], np.shape(T1)[3], np.shape(T1)[4]], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 1], name='LabelData')


# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    dat = {'T1': x, 'T2': x}
    pred = conv_net(dat, num_classes, dropout, reuse=False, is_training=True)
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    loss = tf.reduce_mean(tf.pow(pred - y, 2))
with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # Op to calculate every variable gradient
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    # Op to update all variables according to their gradient
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.metrics.mean_squared_error(labels=y, predictions=pred)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
#tf.summary.scalar("accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
saver = tf.train.Saver()
modelAvailFlag = os.path.isfile(modelPath + '.index')
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Load the model if it exists
    print('attempting to load model')
    if modelAvailFlag:
        saver.restore(sess, modelPath)
        print("Model restored.")
    else:
        print("model file not present, intialising a new model")

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


    # Training cycle
    print('starting training')
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_examples/batch_size)
        print('new epoch')
        # Loop over all batches
        for i in range(total_batch):
            print('batch ', i)
            if i < total_batch:
                batch_xs_T1 = T1[i * batch_size:(i + 1) * batch_size, :, :, :, :]
                batch_xs_T2 = T2[i * batch_size:(i + 1) * batch_size, :, :, :, :]
                batch_ys = label[i*batch_size:(i+1)*batch_size, :]
            else:
                batch_xs_T1 = T1[i * batch_size:, :, :, :, :]
                batch_xs_T2 = T2[i * batch_size:, :, :, :, :]
                batch_ys = label[i*batch_size:, :]


            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([apply_grads, loss, merged_summary_op], feed_dict={x: batch_xs_T1, y: batch_ys})
            print('batch trained, writing to log')
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            print(c)
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        spath = saver.save(sess, modelPath)
        print('model saved at location:  %s' % spath)
    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print("Accuracy:", acc.eval({x: T1, y: label}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=C:\\Users\\Tim\\PycharmProjects\\TensorFlow-Examples\\tensorflow_logs\\example\\ " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
