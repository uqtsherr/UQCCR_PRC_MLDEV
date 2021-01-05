from __future__ import division, print_function, absolute_import


#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import h5py
import numpy as np
#import matplotlib.pyplot as plt
import os.path
import time



fname = '/scratch/medicine/TS-PRC/mat_file_source/FullFeatureDataset.mat'

print(fname)


num_classes = 5
with h5py.File(fname, 'r') as file:
    Dataset = file.get('fullList').value
    if(np.any(np.isnan(Dataset))):
        print("warning, data contains Nan's and needs to be cleaned")



    Features = Dataset[:, 1, 2, :]
    tmp = Features[0, :] -1

    #generate onehot labels matrix
    Labels = np.zeros((np.size(tmp), num_classes))
    Labels[np.arange(np.size(tmp)), tmp.astype(dtype=int)] = 1
    Features = Features[1:, :]

    #pull out all of the channels rather than just one, and flatten
    Features = Dataset[1:, :, 2, :]
    sz = np.shape(Features)
    Features = Features.reshape(sz[0]*sz[1], sz[2])
    if (np.any(np.isnan(Dataset))):
        print("data contains Nan's, removing all erroneous rows")
        Features[~np.isnan(Features).any(axis=1)]
        print("cleaned, shape is:", np.shape(Features))


    for i in range(np.shape(Features)[0]): #Do Normalisation
        max = np.amax(Features[i, :])
        min = np.amin(Features[i, :])
        tmp = Features[i, :] - (max + min) / 2
        tmp = tmp * 2 / (max - min)
        Features[i, :] = tmp





# Training Parameters
learning_rate = 0.001
batch_size = 40000
dropout = 0.8

n_examples = np.shape(Features)[1]
training_epochs = 200
display_epoch = 1
logs_path = '/scratch/medicine/TS-PRC/piglet/tf_logs'
modelPath = '/scratch/medicine/TS-PRC/piglet/model.ckpt'

#Labels = np.expand_dims(Labels, axis=2)
Features = np.transpose(Features, [1, 0])
print('Features shape, Labels shape ', np.shape(Features), ', ', np.shape(Labels))
print('max, min ', np.amax(Features), np.amin(Features))
print('num examples, num classes:',  n_examples, ',  ', num_classes)


# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        print('network layers Features')
        Feats = x_dict['xFeats']
        print(np.shape(Feats))

        # Output layer, class prediction
        fcl = tf.layers.dense(Feats, 120)
        fcl = tf.layers.dense(fcl, 120)
        fcl = tf.layers.dense(fcl, 40)
        tf.nn.dropout(fcl, keep_prob=dropout)
        out = tf.layers.dense(fcl, 5) #for the 5 classes present in the dataset

        return out

# generate placeholder variables to initiate the network model and training functions the features placeholder is n by features length (dim 2)
xFeats = tf.placeholder(tf.float32, [None, np.shape(Features)[1]], name='InputData')

# the labels placeholder is n by 1
y = tf.placeholder(tf.float32, [None, 5], name='LabelData')



# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    dat = {'xFeats': xFeats}
    logits = conv_net(dat, num_classes, dropout, reuse=False, is_training=True)
    prediction = tf.nn.softmax(logits)
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    #loss = tf.reduce_mean(tf.pow(logits - y, 2))
    print(prediction)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y))
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
    acc,update_op = tf.metrics.accuracy(labels=y, predictions=prediction)
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
        total_batch = int(n_examples/batch_size)+1
        permutation = np.random.permutation(n_examples)
        print('new epoch')
        # Loop over all batches
        for i in range(total_batch):
            print('batch ', i)
            if i < total_batch-1:
                set = permutation[i * batch_size:(i + 1) * batch_size]
                batch_xs = Features[set, :]
                batch_ys = Labels[set]
                print('size of batch , labels', np.shape(batch_xs), np.shape(batch_ys))
            else:
                set = permutation[i * batch_size:]
                batch_xs = Features[set, :]
                batch_ys = Labels[set]
                print('got to last batch')
                print('size of batch , labels', np.shape(batch_xs),  np.shape(batch_ys))


            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([apply_grads, loss, merged_summary_op], feed_dict={xFeats: batch_xs, y: batch_ys})
            print('batch trained in ', time.time()-tlast, ' seconds, writing to log')
            tlast = time.time()
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
    _, c, summary = sess.run(update_op, feed_dict={xFeats: Features, y: Labels})

    print("Accuracy:", acc)

    print("Run the command line:\n" \
          "--> tensorboard --logdir=C:\\Users\\Tim\\PycharmProjects\\TensorFlow-Examples\\tensorflow_logs\\example\\ " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
