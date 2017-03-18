'''
Created on Mar 18, 2017
Deep MNISY example
@author: cLennon
'''
##Mostly a repeat of easyMNIST
# #first import the data set and the module.  Data set is stored as numpy arrays, and a function for iterating through as mini-batches is included
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 
# import tensorflow as tf
# #This begins an interactivesession, which may not be as efficient as another types of session
# sess = tf.InteractiveSession()
# 
# #again we will use a softmax model, and these are placeholders for the data
# x = tf.placeholder(tf.float32, shape=[None, 784]) # the none indicates that the number of observations is not specified 
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
# 
# #weights and biases as in the easy example
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
# 
# #initialize variables
# sess.run(tf.global_variables_initializer())
# 
# 
# #Prediction
# y = tf.matmul(x,W) + b
# #define the loss function
# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# 
# #training -- train step are functions that will apply grad descent to the parameters  
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 
# for _ in range(1000):
#     batch = mnist.train.next_batch(100)  # do 100 training examples per iteration
#     train_step.run(feed_dict={x: batch[0], y_: batch[1]}) # feed dict will replace placeholders with data
# #slightly different format than previous one -- why    
# #evaluation
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))





