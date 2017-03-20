'''
Created on Mar 18, 2017
Convolutional neural network from tensorflow tutorial
@author: cLennon
'''
import tensorflow as tf
#Initialize weights with a small amount of noise to break symmerty
#we will use ReLU neurons f(x)=max(x,0), and we initialize with slight positive bias to avoid 'dead neurons'

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # ia am assuming that shape is the mean and the tructation is at 0
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) #here is the afore mentioned positive bias
    return tf.Variable(initial)


 

