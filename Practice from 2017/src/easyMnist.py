import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#describe the input which will eventually hold the mnist images
x = tf.placeholder(tf.float32, [None, 784])


W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#below is soft max classification of the results of our model, which multiplies the 784 x inputs by a variable for each class  
#W is the weights and b is the bias, x is a placeholder for the data
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
#above are the true class values

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#minimize by stoc grad descent with a loss function of cross entropy, defined above
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#launch the session
sess = tf.InteractiveSession()

#initialize variables
tf.global_variables_initializer().run()


#do training

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#checks to make sure the classes are the same
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#average correctness (but what is cast, maybe casting to type float)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

