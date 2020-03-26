from __future__ import division, print_function, absolute_import

import tensorflow as tf
# from tensorflow.contrib import rnn
import tensorflow_hub as hub
import h5py
import math
import numpy as np
# import matplotlib.pyplot as plt
import os.path
from os import listdir
from os.path import isfile, join
import time
import edflib

tf.reset_default_graph()
# def tensorhub variables
classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"  # @param {type:"string"}
IMAGE_SHAPE = (65, 65, 3)

mobilenet = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE, weights=None, include_top=False)

# foldername = 'C:\\Users\\Tim\\EEG Free NAthan'
# foldername = 'C:\\Users\\Tim\\eegNathanTest'
foldername = 'EEGDat'
filelist = [f for f in listdir(foldername) if isfile(join(foldername, f))]
fs = 256
signal_labels = []
signal_nsamples = []


def fileinfo(edf):
    # print("datarecords_in_file", edf.datarecords_in_file)
    # print("signals_in_file:", edf.signals_in_file)

    for ii in range(edf.signals_in_file):
        signal_labels.append(edf.signal_label(ii))
        print("signal_label(%d)" % ii, edf.signal_label(ii), end='')
        print(edf.samples_in_file(ii), edf.samples_in_datarecord(ii), end='')
        signal_nsamples.append(edf.samples_in_file(ii))
        print(edf.samplefrequency(ii))
    return edf.samples_in_file(ii), edf.signals_in_file


def readsignals(edf, nPoints, sig1, buf=None):
    """times in seconds"""
    nsigs = edf.signals_in_file
    for ii in range(nsigs):
        edf.read_digital_signal(ii, 0, nPoints, buf)
        sig1[:, ii] = buf


# takes the label list, produces examples from it
def getLeaveOneOut(LabelsLst, DataLst, N):
    Nrecords = len(DataLst)
    records = []
    for k in range(Nrecords - 1):
        name = DataLst[k][1]
        data = DataLst[k][0]

        annotationNum = int(name.strip(join(foldername, 'eeg')))
        print(annotationNum)
        currLabels = LabelsLst[annotationNum]
        clen = np.shape(currLabels)[0]
        n_chann = np.shape(data)[1]
        print(clen)
        XDat = np.zeros((clen, fs, n_chann))
        for cnt in range(clen - 1):
            timestep = data[cnt * fs:(cnt + 1) * fs][:]
            XDat[cnt, :, :] = timestep

        if k == N:
            validation = (XDat, currLabels)
        else:
            records.append((XDat, currLabels))

    return records, validation


print(foldername)
print(filelist)

# note to Elliot, python indexing starts at zero not 1
num_classes = 2
annotations = []
i = 0
if isfile(join(foldername, 'annotations_EEGdata.mat')):
    with h5py.File(join(foldername, 'annotations_EEGdata.mat')) as file:
        for c in file['annotat_new']:
            for r in range(len(c)):
                annotations.append(file[c[r]][()])
                i = i + 1
print(annotations)

eegdata = []
for file in filelist:
    fpath = join(foldername, file)
    p, ext = os.path.splitext(fpath)
    if (ext == '.edf'):
        print(fpath + " is a .edf file")
        edf = edflib.EdfReader(fpath)
        samples, nSigs = fileinfo(edf)
        sig1 = np.zeros((samples, nSigs), dtype='int32')
        buf = np.zeros(samples, dtype='int32')
        readsignals(edf, samples, sig1, buf)
        datName = (sig1, p)
        eegdata.append(datName)
        print(sig1)
    else:
        print(fpath + " is not a .edf file")

records, validation = getLeaveOneOut(annotations, eegdata, 0);

##End data load section - records is a tuple of labels,Dataset
# Data is currently imported as a record, where records are made up of a (label data) tuple. label file of N x 3 samples where N is record length in seconds
# The a data is matrix of 21x (NxFs) where fs is 256Hz

# Training Parameters
learning_rate = 0.001
batch_size = 100
dropout = 0.2

n_recs = np.shape(records)[0]
print(n_recs)
training_epochs = 5
display_epoch = 1
num_hidden = 40
timesteps = 256
channels = 21

# dfefine logging paths

logs_path = 'C:\\TF\\tf_logs\\'
modelPath = 'C:\\TF\\Models\\RNNforEEG\\'

logs_path = 'TF\\tf_logs\\'
modelPath = 'TF\\Models\\RNNforEEG\\'


# Create the neural network
def dense_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        print('network layers Features')
        Feats = x_dict['xFeats']
        Feats = tf.transpose(Feats, perm=[0, 2, 1])
        # channs = tf.unstack(Feats, 2)
        stfts = tf.contrib.signal.stft(Feats, 128, 2, 128)
        # stfts = tf.cast(stfts, tf.complex64)
        stfts = tf.abs(tf.transpose(stfts, [0, 2, 3, 1])) / 256
        # stfts = tf.image.per_image_standardization(stfts)
        channs = tf.unstack(stfts, axis=3)

        # process images in stacks of 3 (stacking the stfts into feaux RGB imgs)
        # need to reduce the size of each substack, use a dense connect net for each subsection
        subimg1 = tf.stack([channs[0], channs[1], channs[2]], axis=3)
        t1 = mobilenet(subimg1)
        t1 = tf.reshape(t1, [-1, 3 * 3 * 1280])
        t1 = tf.layers.dense(t1, 1000)
        t1 = tf.nn.dropout(t1, keep_prob=0.85)
        subimg2 = tf.stack([channs[3], channs[4], channs[5]], axis=3)
        t2 = mobilenet(subimg2)
        t2 = tf.reshape(t2, [-1, 3 * 3 * 1280])
        t2 = tf.layers.dense(t2, 1000)
        t2 = tf.nn.dropout(t2, keep_prob=0.85)
        subimg3 = tf.stack([channs[6], channs[7], channs[8]], axis=3)
        t3 = mobilenet(subimg3)
        t3 = tf.reshape(t3, [-1, 3 * 3 * 1280])
        t3 = tf.layers.dense(t3, 1000)
        t3 = tf.nn.dropout(t3, keep_prob=0.85)
        subimg4 = tf.stack([channs[9], channs[10], channs[11]], axis=3)
        t4 = mobilenet(subimg4)
        t4 = tf.reshape(t4, [-1, 3 * 3 * 1280])
        t4 = tf.layers.dense(t4, 1000)
        t4 = tf.nn.dropout(t4, keep_prob=0.85)
        subimg5 = tf.stack([channs[12], channs[13], channs[14]], axis=3)
        t5 = mobilenet(subimg5)
        t5 = tf.reshape(t5, [-1, 3 * 3 * 1280])
        t5 = tf.layers.dense(t5, 1000)
        t5 = tf.nn.dropout(t5, keep_prob=0.85)
        subimg6 = tf.stack([channs[15], channs[16], channs[17]], axis=3)
        t6 = mobilenet(subimg6)
        t6 = tf.reshape(t6, [-1, 3 * 3 * 1280])
        t6 = tf.layers.dense(t6, 1000)
        t6 = tf.nn.dropout(t6, keep_prob=0.85)
        subimg7 = tf.stack([channs[18], channs[19], channs[20]], axis=3)
        t7 = mobilenet(subimg7)
        t7 = tf.reshape(t7, [-1, 3 * 3 * 1280])
        t7 = tf.layers.dense(t7, 1000)
        t7 = tf.nn.dropout(t7, keep_prob=0.85)

        # dense connect the outputs
        tmplayer = tf.concat([t1, t2, t3, t4, t5, t6, t7], axis=1)
        tmplayer = tf.reshape(tmplayer, [-1, 1000 * 7])
        # Output layer, class prediction
        fcl = tf.layers.dense(tmplayer, 3072)
        fcl = tf.nn.dropout(fcl, keep_prob=0.7)
        fcl = tf.layers.dense(fcl, 1000)
        tf.nn.dropout(fcl, keep_prob=0.6)
        out = tf.layers.dense(fcl, 2)  # for the 2 classes present in the dataset

        return out


def BiRNN(x_dict, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)
    x = x_dict['xFeats']
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    x = tf.reshape(x, [-1, timesteps * channels])
    x = tf.unstack(x, timesteps * channels,
                   1)  # this is the major hurdle. rearranging the input data to match the intended input space.

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                 dtype=tf.float32)
    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                     dtype=tf.float32)
    except Exception:  # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                               dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


print(np.shape(records[0][0]))
# generate placeholder variables to initiate the network model and training functions the features placeholder is n by features length (dim 2)
xFeats = tf.placeholder(tf.float32, [None, np.shape(records[0][0])[1], np.shape(records[0][0])[2]], name='InputData')

# the labels placeholder is n by 1
y = tf.placeholder(tf.float32, [None, 2], name='LabelData')

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    dat = {'xFeats': xFeats}
    # logits = BiRNN(dat, weights, biases)
    logits = dense_net(dat, num_classes, dropout, reuse=False, is_training=True)
    prediction = tf.nn.softmax(logits)
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    # loss = tf.reduce_mean(tf.pow(logits - y, 2))
    print(prediction)
    weightedLoss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y)
    loss = tf.reduce_mean(weightedLoss)
with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # Op to calculate every variable gradient
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    # Op to update all variables according to their gradient
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

with tf.name_scope('Accuracy'):
    # Accuracy
    yh = tf.where(tf.equal(y, 1))[:, 1]
    print(y)
    ph = tf.arg_max(prediction, 1)
    print(prediction)
    acc, acc_op = tf.metrics.accuracy(labels=yh, predictions=ph)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", tf.squeeze(acc))
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
saver = tf.train.Saver()
modelAvailFlag = os.path.isfile(modelPath + '.index')
# Start training

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    # Load the model if it exists
    print('attempting to load model')
    if modelAvailFlag:
        # try restore model
        try:
            saver.restore(sess, modelPath)
            print("Model restored.")
        except:
            print("model is broken, likely the network structure has changed. will overwrite old model")
    else:
        print("model file not present, intialising a new model")

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    tlast = time.time()
    # Training cycle
    print('starting training')
    for epoch in range(training_epochs):
        avg_cost = 0.
        # print('new epoch')
        # Loop over all batches
        i = 0;
        for rec in records:
            per = np.random.permutation(np.shape(rec[0])[0])
            xrec = rec[0][per, :, :]
            yrec = rec[1][per, :]

            for i in range(math.floor(np.shape(rec[0])[0] / batch_size)):
                # print('batch ', i)
                batch_xs = xrec[batch_size * i:batch_size * (i + 1), :, :]
                batch_ys = yrec[batch_size * i:batch_size * (i + 1), 1]  # grab expert 1's markup labels
                batch_ys = tf.one_hot(batch_ys, 2).eval()  # get one hot tensor and return to a numpy vector
                # print('size of batch , labels', np.shape(batch_xs), np.shape(batch_ys))

                # Run optimization op (backprop), cost op (to get loss value)
                # and summary nodes
                _, c, summary = sess.run([apply_grads, loss, merged_summary_op],
                                         feed_dict={xFeats: batch_xs, y: batch_ys})
                # print('batch trained in ', time.time()-tlast, ' seconds, writing to log')
                tlast = time.time()
                # Write logs at every iteration
                summary_writer.add_summary(summary, epoch * n_recs + i)
                # Compute average loss
                print(c)
                avg_cost += c / math.floor(np.shape(rec[0])[0] / batch_size)
            print("average cost of record is:  ", "{:.9f}".format(avg_cost))
        # Display logs per epoch step
        if (epoch + 1) % display_epoch == 0:
            xF = []
            yF = []
            accin = sess.run(acc_op, feed_dict={xFeats: batch_xs, y: batch_ys})
            perm = np.random.permutation(np.shape(validation[0])[0])[0:110]
            accout = sess.run(acc_op, feed_dict={xFeats: validation[0][perm, :, :],
                                                 y: tf.one_hot(validation[1][perm, 1], 2).eval()})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost),
                  " internal accuracy; validation accuracy", accin, accout)
        spath = saver.save(sess, modelPath)
        print('model saved at location:  %s' % spath)
    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    acc = sess.run(acc_op,
                   feed_dict={xFeats: validation[0][perm, :, :], y: tf.one_hot(validation[1][perm, 1], 2).eval()})
    pred = sess.run(prediction,
                    feed_dict={xFeats: validation[0][perm, :, :], y: tf.one_hot(validation[1][perm, 1], 2).eval()})
    print("Accuracy:", acc)
    print(pred)
    np.savetxt("accuracyRNNforEEG.csv", acc, delimiter=",")
    np.savetxt("predictionRNNforEEG.csv", pred, delimiter=",")

    print("Run the command line:\n" \
          "--> tensorboard --logdir=C:\\TF\\tf_logs\\ " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")