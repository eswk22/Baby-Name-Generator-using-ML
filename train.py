from __future__ import print_function
import numpy as np
import tensorflow as tf
import math as math
import os
import string
import time
from ReadData import ReadData
import logging

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

# Target log path
logs_path =  os.path.dirname(os.path.realpath(__file__)) + "\\logs"
writer = tf.summary.FileWriter(logs_path)

def build_dataset():
    dictionary = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    return dictionary

def convert(word):
    dict1 = {value: (int(key) + 1) for key, value in enumerate(list(string.ascii_lowercase))}
    return [str(dict1[alp.lower()]) for alp in list(tf.string_split(word, delimiter="").values.eval())]

def RNN(x,weights,biases):
    # reshape to [1, n_input]
    #x = tf.reshape(x, [len(x),-1])
    #x = tf.split(x,n_input,1)
    # print(x)
    lstm = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    rnn_cell = tf.contrib.rnn.MultiRNNCell([lstm,lstm])

    # generate prediction
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

dictionary = build_dataset()

# Parameters
learning_rate = 0.001
batch_size = 100
epochs = 10

n_input = 1
n_hidden = 24

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, len(dictionary)])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, len(dictionary)]))
}
biases = {
    'out': tf.Variable(tf.random_normal([len(dictionary)]))
}

def main():
    readdata = ReadData()

    trainingFiles,testingFiles = readdata.filePathConstructor()
    features = readdata.input_pipeline(trainingFiles,batch_size)

    with tf.Session() as sess:
        # Create the graph, etc.
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        dict1 = {value: (int(key) + 1) for key, value in enumerate(list(string.ascii_lowercase))}
        dict1[' '] = 0 
        dict1[';'] =-1
        dict1['-'] =-1
        vocab_size = len(dict1)
        for i in range(1): 
            example_batch = tf.reshape(features,[-1])
            item = tf.string_split(example_batch, delimiter="").values.eval()
            chars = [dict1[alp.decode().lower()] for alp in list(item)]
            data_size = len(chars)
            print('Data has %d characters, %d unique.' % (data_size, vocab_size))
            
            # # Hyper-parameters
            # hidden_size   = 100  # hidden layer's size
            # seq_length    = 25   # number of steps to unroll
            # learning_rate = 1e-1
            
            
            # inputs     = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32, name="inputs")
            # targets    = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32, name="targets")
            # init_state = tf.placeholder(shape=[1, hidden_size], dtype=tf.float32, name="state")

            # intializer = tf.random_normal_initializer(stddev=1.0)

            # with tf.variable_scope("RNN") as scope:
            #     hs_t = init_state
            #     ys = []
            #     for t,xs_t in enumerate(tf.split(inputs,seq_length,axis=0)):
            #         if t > 0:scope.reuse_variables()
            #         Wxh = tf.get_variable("Wxh",shape=[vocab_size,hidden_size],dtype=tf.float32,intializer=intializer)
            #         Whh = tf.get_variable("Whh",shape=[hidden_size,hidden_size],dtype=tf.float32,intializer=intializer)
            #         Why = tf.get_variable("Why",shape=[hidden_size,vocab_size],dtype=tf.float32,intializer=initializer)
            #         bh = tf.get_variable("bh",shape=[hidden_size],intializer=intializer)
            #         by = tf.get_variable("by",shape=[vocab_size],initializer=intializer)

            #         hs_t = tf.tanh(tf.matmul(xs_t,Wxh) + tf.matmul(hs_t,Whh) + bh)
            #         ys_t = tf.matmul(hs_t,Why) + by
            #         ys.append(ys_t)

            # h_prev = hs_t
            
            # output_softmax = tf.nn.softmax(ys[-1])

            # outputs = tf.concat(ys,axis=0)
            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets,logits=outputs))

            # #optimizer
            # minimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # grad_and_vars = minimizer.compute_gradients(loss)



            # pred = RNN(chars,weights,biases)
            # # Loss and optimizer
            # # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
            # # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

            # # # Model evaluation
            # # correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
            # # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            # # print(example_batch) 
       
        

        

        coord.request_stop() 
        coord.join(threads)

if __name__ == '__main__':
    tf.logging._logger.setLevel(logging.INFO)
    main()

