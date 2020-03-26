from __future__ import division, print_function, absolute_import


import tensorflow as tf
from tensorflow.contrib import rnn
import h5py
import numpy as np
#import matplotlib.pyplot as plt
import os.path
from os import listdir
from os.path import isfile, join, isdir, splitext
import time
import edflib

foldername = 'C:\\piglet labelled eeg\\'
folderlist = [join(foldername, f) for f in listdir(foldername) if isdir(join(foldername, f))]
filelist = []
for fold in folderlist:
    fi = [join(fold, f) for f in listdir(fold) if isfile(join(fold, f))]
    filelist = filelist + fi

print(filelist)

fs =256
signal_labels = []
signal_nsamples = []
Records = []
for f in filelist:
    a = splitext(f)
    if a[1] == '.mat':

        with h5py.File(f) as file:
            print(file['Records'].keys())
            Class = file['Records']['ClassNum'].value
            print(Class)
            xTFDs = file['Records']['TFDs'][()]
            xEEG = file['Records']['eeg']
            xFeatures = file['Records']['features']

print(foldername)
print(filelist)

#note to Elliot, python indexing starts at zero not 1
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
    fpath = join(foldername,file)
    p, ext = os.path.splitext(fpath)
    if(ext == '.edf'):
        print(fpath + " is a .edf file")
        edf = edflib.EdfReader(fpath)
        samples, nSigs = fileinfo(edf)
        sig1 = np.zeros((samples,nSigs), dtype='int32')
        buf =  np.zeros(samples, dtype='int32')
        readsignals(edf,samples,sig1,buf)
        datName = (sig1,p)
        eegdata.append(datName)
        print(sig1)
    else:
        print(fpath + " is not a .edf file")


records, validation = getLeaveOneOut(annotations, eegdata, 0);



##End data load section - records is a tuple of labels,Dataset
#Data is currently imported as a record, where records are made up of a (label data) tuple. label file of N x 3 samples where N is record length in seconds
#The a data is matrix of 21x (NxFs) where fs is 256Hz

# Training Parameters
learning_rate = 0.001
batch_size = 40000
dropout = 0.8

n_recs = np.shape(records)[0]
print(n_recs)
training_epochs = 3
display_epoch = 1
num_hidden = 40
timesteps = 256
channels = 21

#dfefine logging paths
logs_path = 'S:\\UQCCR-Colditz\\Signal Processing File Sharing\\For Elliot\\Tensorflow_logs\\example2\\'
modelPath = 'S:\\UQCCR-Colditz\\Signal Processing File Sharing\\For Elliot\\Tensorflow\\PigletEEG\\model.ckpt'


logs_path = 'C:\\TF\\tf_logs\\'
modelPath = 'C:\\TF\\Models\\RNNforEEG\\'

# Create the neural network
def dense_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        print('network layers Features')
        Feats = x_dict['xFeats']
        Feats = tf.transpose(Feats, perm=[0,2,1])
        channs = tf.unstack(Feats, 2)
        stfts = tf.signal.stft(channs,64,8,64)
        stfts = tf.cast(stfts, tf.complex64)
        tmplayer = tf.layers.conv2d(stfts, 32, 10, activation=tf.nn.relu,data_format='channels_last')
        print(np.shape(tmplayer))
        tmplayer = tf.layers.conv2d(tmplayer, 32, 10, activation=tf.nn.relu)
        print(np.shape(tmplayer))
        tmplayer = tf.reshape(tmplayer, [-1,3*32])
        # Output layer, class prediction
        fcl = tf.layers.dense(tmplayer, 500)
        fcl = tf.layers.dense(fcl, 200)
        fcl = tf.layers.dense(fcl, 100)
        fcl = tf.layers.dense(fcl, 40)
        tf.nn.dropout(fcl, keep_prob=dropout)
        out = tf.layers.dense(fcl, 2) #for the 5 classes present in the dataset

        return out

def BiRNN(x_dict, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)
        x = x_dict['xFeats']
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
        x = tf.reshape(x, [-1,timesteps*channels])
        x = tf.unstack(x, timesteps*channels, 1)        #this is the major hurdle. rearranging the input data to match the intended input space.

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
print(np.shape(records[0][0]) )
# generate placeholder variables to initiate the network model and training functions the features placeholder is n by features length (dim 2)
xFeats = tf.placeholder(tf.float32, [None, np.shape(records[0][0])[1], np.shape(records[0][0])[2]], name='InputData')

# the labels placeholder is n by 1
y = tf.placeholder(tf.float32, [None, 2], name='LabelData')

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
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
    #loss = tf.reduce_mean(tf.pow(logits - y, 2))
    print(prediction)
    weightedLoss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y)
    loss = tf.reduce_mean(weightedLoss)
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
    acc,acc_op = tf.metrics.accuracy(labels=y, predictions=prediction)
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
        #try restore model
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
        #print('new epoch')
        # Loop over all batches
        i = 0;
        for rec in records:
            i = i+1;
            #print('batch ', i)
            batch_xs = rec[0]
            batch_ys = rec[1][:, 1]
            batch_ys = tf.one_hot(batch_ys,2).eval()  #get one hot tensor and return to a numpy vector
            #print('size of batch , labels', np.shape(batch_xs), np.shape(batch_ys))



            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([apply_grads, loss, merged_summary_op], feed_dict={xFeats: batch_xs, y: batch_ys})
            #print('batch trained in ', time.time()-tlast, ' seconds, writing to log')
            tlast = time.time()
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * n_recs + i)
            # Compute average loss
            print(c)
            avg_cost += c / n_recs
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            xF = []
            yF = []
            accin = sess.run(acc_op, feed_dict={xFeats: batch_xs, y: batch_ys})
            accout = sess.run(acc_op, feed_dict={xFeats: validation[0], y: tf.one_hot(validation[1][:, 1], 2).eval()})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), " internal accuracy; validation accuracy", accin, accout)
        spath = saver.save(sess, modelPath)
        print('model saved at location:  %s' % spath)
    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    acc = sess.run(acc_op, feed_dict={xFeats: validation[0], y: tf.one_hot(validation[1][:, 1], 2).eval()})
    pred = sess.run(prediction, feed_dict={xFeats: validation[0], y: tf.one_hot(validation[1][:, 1], 2).eval()})
    print("Accuracy:", acc)
    print(pred)
    np.savetxt("foo.csv", pred, delimiter=",")

    print("Run the command line:\n" \
          "--> tensorboard --logdir=C:\\Users\\Tim\\PycharmProjects\\TensorFlow-Examples\\tensorflow_logs\\example\\ " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
