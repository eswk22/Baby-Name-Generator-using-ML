from __future__ import print_function
import tensorflow as tf
import numpy
import string
from ReadData import ReadData

#utils

def builddictionary():
     dict1 = {value: (int(key) + 1) for key, value in enumerate(list(string.ascii_lowercase))}
     dict1[' '] = 0 
     dict1[';'] =-1
     dict1['-'] =-1
     dict1[','] =-1
     return dict1

def getInputData(batchsize):
    readdata = ReadData()
    trainingFiles,testingFiles = readdata.filePathConstructor()
    features = readdata.input_pipeline(trainingFiles,batchsize)
    example_batch = tf.reshape(features,[-1])
    item = tf.string_split(example_batch, delimiter="").values.eval()
    return  [dict1[alp.decode().lower()] for alp in list(item)]
    #return tf.one_hot(data,len(dict1))

# Parameters

learningrate = 0.001
batch_size = 1
training_steps = 1
display_step = 200

dict1 = builddictionary()

# Network Parameters
n_hidden = 100
seq_len = 29
n_input = len(dict1)
n_class = len(dict1)

# # Weight & Biases
# W = tf.Variable(initial_value=tf.random_normal([n_hidden,n_class]),trainable=True)
# b = tf.Variable(initial_value=tf.random_normal([n_class]),trainable=True)

# # Placeholders for input
# X = tf.placeholder(dtype=tf.float32,shape=[None,len(dict1),seq_len],name="x")
# Y = tf.placeholder(dtype=tf.float32,shape=[None,len(dict1)],name="y")

# # Building Model
# def drnn(x,weights,biases):
#     x = tf.unstack(x,seq_len,1)
#     lstm = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
#     outputs = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32)

#     # Linear activation, using rnn inner loop last output
#     return tf.matmul(outputs[-1], weights) + biases    

# # Call Model
# logits = drnn(X,W,b)
# predict = tf.nn.softmax(logits)
# Y = predict
# # Cost & Optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)
# train_op = optimizer.minimize(loss_op)

# # build accuracy
# correct_pred = tf.equal(tf.argmax(predict,1),tf.argmax(Y,1))
# accuracy = tf.reduce_mean(correct_pred,tf.float32)

# Initiate Global variable
init = tf.global_variables_initializer()

readdata = ReadData()
trainingFiles,testingFiles = readdata.filePathConstructor()
features = readdata.input_pipeline(trainingFiles,batch_size)

# Start training
with tf.Session() as sess:
    # init session
    sess.run(init)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
        
    # loop training steps
    for step in range(training_steps):
        # read input data
        example_batch = tf.reshape(features,[-1])
        item = tf.string_split(example_batch, delimiter="").values.eval()
        batch_x = [dict1[alp.decode().lower()] for alp in list(item)]
        #batch_x = getInputData(batch_size)
        batch_x = tf.reshape([batch_x,seq_len,seq_len],[-1])
        print(batch_x)
        # Pass input to Optimizer
        #sess.run(train_op,feed_dict={X:batch_x})
        # Pass input to accuracy if we need to print
        # if step % display_step == 0 :
        #     loss,acc = sess.run([loss_op,accuracy],feed_dict={X:batch_x})
        #     print("Step " + str(step) + ", Minibatch Loss= " + \
        #           "{:.4f}".format(loss) + ", Training Accuracy= " + \
        #           "{:.3f}".format(acc))
    coord.request_stop() 
    coord.join(threads)

    print("Optimization finished")    