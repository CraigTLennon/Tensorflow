'''
Created on Mar 18, 2017
Convolutional neural network from tensorflow tutorial
@author: cLennon
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Initialize weights with a small amount of noise to break symmerty
#we will use ReLU neurons f(x)=max(x,0), and we initialize with slight positive bias to avoid 'dead neurons'

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # ia am assuming that shape is the mean and the tructation is at 0
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) #here is the afore mentioned positive bias
    return tf.Variable(initial)

#convolution filter size is the dim of the filter, e.g 3X3 pixels (cells) 
#strides are the number of cells over the filter moves at each step
# padding adds zeros around the edges of the image.
#ksize is the size of the window for each dimension of the input tensor, e.g for a 64x64 pixel image, rgb , ksize=[batch_size, 64, 64, 3]
#if you have a 2X2 window over which you take max, it would be [1,2,2,1], with 1's for batch size and channel bc we want max for single example and 1 channel

 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#OUr first layer will be convolution followed by max pooling.  Note mnist inages start as 28X28 pixels
#For each 5 by 5 patch, the convolution will compute 32 features (why 32 -- where did this come from)
# there is also bias vector with a component for each output channel (each of 32 features)



x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#We need to reshape x to be a 4d tensor, with second and third dims corresponding to image width and height.
x_image = tf.reshape(x, [-1,28,28,1]) #what is -1 doing here.  Why not 1?

#now convolve x_image with weight tensor
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
