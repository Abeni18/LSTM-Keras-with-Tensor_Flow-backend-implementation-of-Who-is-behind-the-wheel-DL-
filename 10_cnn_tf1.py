# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 14:05:56 2018

@author: Abenezer
"""

import numpy as np
import os 
from util.utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder



def pad_along_axis(array: np.ndarray, target_length, axis=0):

    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return a

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b

def get_batches(X, y, batch_size = 100):
	""" Return a generator for batches """
	n_batches = len(X) // batch_size
	X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

	# Loop over batches and yield
	for b in range(0, len(X), batch_size):
            yield X[b:b+batch_size], y[b:b+batch_size]
            
#def get_batches2(train_x, train_y, batch_size=100):
#    n_batches = len(train_x) // batch_size
#    
#    for b in range (n_batches):
#        offset = (b * batch_size) % (train_y.shape[0] - batch_size)
#        batch_x = train_x[offset:(offset + batch_size), :, :, :]
#        batch_y = train_y[offset:(offset + batch_size), :]
#        return batch_x, batch_y
#    

def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)
    
def plot_activity(activity,data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize = (15, 10), sharex = True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()
    

data = pd.read_csv('KIA1.csv')

data = data.drop(columns=['Torque_scaling_factor(standardization)', 'Target_engine_speed_used_in_lock-up_module'])
#X = data.iloc[:,:48].values
#y = data.iloc[:,49].values

X = data.iloc[:,:-1].values
y = data.iloc[:,47 ].values.reshape(-1,1)

' Encoding Catagorical Data and OneHotEncoding'

# label ecoding y
labelencoder_y = LabelEncoder()
y  = labelencoder_y.fit_transform(y)


'''

for i in range(int(len(X[1]))):
    plt.figure(i)
    plt.plot(X[:1000,i], label=data.columns[i])
    plt.legend()


for i in ([6,21,22,26,48]):
    plt.figure(i)
    plt.plot(X[:,i], label=data.columns[i])
    plt.legend()
    plt.show()

for i in ([22,26]):
    plt.figure(i)
    plt.plot(X[67000:66000,i], label=data.columns[i])
    plt.legend()
    plt.show()

'''


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle =True)

'Train/validiation split'

X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, y_train, random_state = 101, shuffle =False)




' Encoding Catagorical Data and OneHotEncoding'

#y = np.reshape(y,(-1,1))

# onehot endcoding x
#onehotencoder = OneHotEncoder(categorical_features = [46])
#X_tr = onehotencoder.fit_transform(X_tr).toarray()






y_test = y_test.reshape(-1,1)
y_tr = y_tr.reshape(-1,1)
y_vld = y_vld.reshape(-1,1)



'Normalizeation'
scale_X = StandardScaler()
X_tr = scale_X.fit_transform(X_tr)
X_vld = scale_X.transform(X_vld)
X_test = scale_X.transform(X_test)




' Encoding Catagorical Data and OneHotEncoding'

onehotencodery = OneHotEncoder(categorical_features = [0])
y_test = onehotencodery.fit_transform(y_test).toarray()
y_tr = onehotencodery.fit_transform(y_tr).toarray()
y_vld = onehotencodery.fit_transform(y_vld).toarray()

dim = int(input('Sequence length:  '))

import tensorflow as tf

'Hyper parameters'
batch_size = 640
seq_len = dim
learning_rate = 0.0001
epochs = int(input('Number of epochs:  '))

n_classes = 10
n_channels = int(X_tr.shape[1])



graph = tf.Graph()

# construct placeholders

with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name='inputs')
    labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name= 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')


'Build CNN'

with graph.as_default():
    '(batch, 16,47) --> (batch, 8, 18)'    
    conv1 = tf.layers.conv1d(inputs = inputs_, filters=15, kernel_size=11, strides=1, padding='same', activation=tf.nn.relu, name='Convolution_Layer_1')
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=1, strides=1, padding='same', name='max_pooling_1')
    
    '''    
    '(batch, 8, 18) --> (batch, 4, 36)'
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters= 36, kernel_size=2, strides=1, padding='same', activation = tf.nn.relu, name='Convolution_Layer_2')
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same', name='max_pooling_2')
    
    'Conv3  (batch, 4, 36) --> (batch, 2, 72) '
    conv3 = tf.layers.conv1d(inputs = max_pool_2, filters = 72, kernel_size=2, strides=1, padding='same', activation = tf.nn.relu, name='Convolution_Layer_3' )
    max_pool_3 = tf.layers.max_pooling1d(inputs = conv3, pool_size=2, strides=2, padding='same', name='max_pooling_3')
    
    'Conv4  (batch, 2, 72) --> (batch, 1, 144) '
    conv4 = tf.layers.conv1d(inputs = max_pool_3, filters = 144, kernel_size = 2, strides=1, padding = 'same', activation = tf.nn.relu, name='Convolution_Layer_4')
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same', name='max_pooling_4')
    '''
    
    
'Fully Conected NN (flatten out to pass it to the classifier)'

with graph.as_default():
    'Flatten and add dropout'

    flat = tf.reshape(max_pool_1, (-1,  15)) 
    #flat = max_pool_1
    flat = tf.nn.dropout(flat, keep_prob = keep_prob_)
    
    'Prediction '
    prediction = tf.layers.dense(flat, n_classes)
    
    
    'cost function and optimizer'
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels = labels_ ))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_).minimize(cost)
    
    'Accuracy '
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name= 'accuracy')
    
    
    
    
"Data set preparation"

#with graph.as_default:
    
#dataset = tf.data.Dataset.from_tensor_slices((inputs_, labels_)).batch(batch_size).repeat()
#dataset = dataset.batch(batch_size)
#iterator = dataset.make_initializable_iterator()

    
#    data_X, data_y = iterator.get_next()
#    data_y = tf.cast(data_y, tf.int32)
#    model 
    
'''
vld_dataset = tf.data.Dataset.from_tensor_slices((X_vld, y_vld))
    
'Generate the comple dataset required in teh pipline'
tr_dataset = tr_dataset.repeat(epochs).batch(batch_size)
tr_iterator = tr_dataset.make_one_shot_iterator()

vld_dataset = vld_dataset.repeat(epochs).batch(batch_size)
vld_iterator = vld_dataset.make_one_shot_iterator()


X_tr_b, y_tr_b = tr_iterator.get_next()
x_vld_b, y_vld_b = vld_iterator.get_next()
'''
    
     
    
    
    
if not os.path.exists('summaries'):
    os.mkdir('summaries')
    
if not os.path.exists(os.path.join('summaries', 'first')):
    os.mkdir(os.path.join('summaries','first'))
    
    
validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.train.Saver()
    

with tf.Session(graph= graph) as sess:
    
    sess.run(tf.global_variables_initializer())
    iteration = 1
    
    
    
    
        
    '''    
        
        total_batches = len(X_tr) // batch_size
        for epoch in range (epochs):
            for b in range (total_batches):
                offset = (b * batch_size) % (y_tr.shape[0] - batch_size)
                x = X_tr[offset:(offset + batch_size), :]
                y = y_tr[offset:(offset + batch_size), :]
                
                feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, learning_rate_ : learning_rate}
                
                # Loss
                loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
                train_acc.append(acc)
                train_loss.append(loss)
              
                # Print at each 5 iters
                if (iteration % 5 == 0):
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "Train loss: {:6f}".format(loss),
                          "Train acc: {:.6f}".format(acc))
        
        
        
             
        
        
    '''
      
        
        
    'Loop over epochs ' 
    
    for e in range(epochs):
        
        # Loop over batches
#        for x,y in get_batches(X_tr, y_tr, batch_size):
            
#        x_b, y_b = tf.train.batch( [X_tr,y_tr], batch_size = 100 )
        #for x,y in iterator.get_next(100):
        
        
              
               
        
        for x,y in get_batches(X_tr, y_tr, batch_size):    
            
            
            x = x.reshape(-1,dim,47)
            
            #y = np.argmax(y, axis = 1)
            yy = []
            for i in range(640//dim):
                offset = (i * dim) % (640 - dim)
                y1 = y[offset:(offset + dim), :]
#                y = (y[i: i + i*dim,])
               
                y2 = np.argmax(y1, axis=1)
                yy.append(np.argmax(y2))
                
            yy = np.reshape(yy, (-1,1))
            yy = onehotencodery.fit_transform(yy).toarray()
            yy = pad_along_axis(yy, 10, axis=1)
            
#            counts = np.bi
#                y = y.reshape(-1,10)
            
            # Feed dictionary
            feed = {inputs_ : x, labels_ : yy, keep_prob_ : 0.5, learning_rate_ : learning_rate}
            
            # Loss
            loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
            
            train_acc.append(acc)
            train_loss.append(loss)
            #print('started')
            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))
            
            # Compute validation loss at every 10 iterations
            if (iteration%10 == 0):                
                val_acc_ = []
                val_loss_ = []
                
                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    
                    
                    x_v = x_v.reshape(-1,dim,47)
            
                    #y = np.argmax(y, axis = 1)
                    yy_v = []
                    for i in range(640//dim):
                        offset = (i * dim) % (640 - dim)
                        y1_v = y_v[offset:(offset + dim), :]
        #                y = (y[i: i + i*dim,])
                       
                        y2_v = np.argmax(y1_v, axis=1)
                        yy_v.append(np.argmax(y2_v))
                        
                    yy_v = np.reshape(yy_v, (-1,1))
                    yy_v = onehotencodery.fit_transform(yy_v).toarray()
                    yy_v = pad_along_axis(yy_v, 10, axis=1)
                    
                    
                    
                    #y_v = y_v.reshape(-1,10)
                    # Feed
                    feed = {inputs_ : x_v, labels_ : yy_v, keep_prob_ : 1.0}  
                    
                    # Loss
                    loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)                    
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
                
                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                
                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))
        
            # Iterate 
            iteration += 1
    
    saver.save(sess,"summaries/har.ckpt")
    
t = np.arange(iteration-1)

plt.figure(figsize = (6,6))
plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(validation_loss), 'b*')
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
    

# Plot Accuracies
plt.figure(figsize = (6,6))

plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], validation_acc, 'b*')
plt.xlabel("iteration")
plt.ylabel("Accuray")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
    
    


test_acc = []

with tf.Session(graph=graph) as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint('summaries'))
    
    for x_t, y_t in get_batches(X_test, y_test, batch_size):
    
        x_t = x_t.reshape(-1,dim,47)
    
        #y = np.argmax(y, axis = 1)
        yy_t = []
        for i in range(640//dim):
            offset = (i * dim) % (640 - dim)
            y1_t = y_t[offset:(offset + dim), :]
    #                y = (y[i: i + i*dim,])
           
            y2_t = np.argmax(y1_t, axis=1)
            yy_t.append(np.argmax(y2_t))
            
        yy_t = np.reshape(yy_t, (-1,1))
        yy_t = onehotencodery.fit_transform(yy_t).toarray()
        yy_t = pad_along_axis(yy_t, 10, axis=1)
        
        
        
        #y_t = y_t.reshape(-1,10)
                    
        feed = {inputs_: x_t,
                labels_: yy_t,
                keep_prob_: 1}
        
        batch_acc = sess.run(accuracy, feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.6f}".format(np.mean(test_acc)))




