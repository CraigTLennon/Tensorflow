import tensorflow as tf
import numpy as np

#create constants which cannot change value
# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0) # also tf.float32 implicitly
#print(node1, node2)

#print(sess.run([node1,node2]))

#add together those two constants
# node3 = tf.add(node1, node2)
#print("node3: ", node3)
#print("sess.run(node3): ",sess.run(node3))
# a = tf.placeholder(tf.float32)
#b = tf.placeholder(tf.float32)
#adder_node = a + b  # + provides a shortcut for tf.add(a, b)
#print(sess.run(adder_node, {a: 3, b:4.5}))
#print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
#add_and_triple = 3.1 * adder_node 
#print(sess.run(add_and_triple, {a: 3, b:4.5}))


#introducing variables which can be used for training.  These need to be initialized with the global_variables initializer
# W = tf.Variable([.3], tf.float32)
# b = tf.Variable([-.3], tf.float32)
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b
# # #linear_model= tf.multiply(W, x)
# init = tf.global_variables_initializer()
# 
# 
# 
# 
# #variabls can be fixed to specific values if required
# #fixW = tf.assign(W, [-1.])
# #fixb = tf.assign(b, [1.])
# #sess.run([fixW, fixb])
# #print(sess.run(linear_model, {x:[1.0,2.0,3.0,4.0]}))
# 
# #Now to test a linear regression model, defining a dependent variable, y, and a loss function (squared error), and 
# y = tf.placeholder(tf.float32)
# squared_deltas = tf.square(linear_model - y)
# loss = tf.reduce_sum(squared_deltas)
# 
# #optimizers slowly change variables to minimize the loss function.  The simplest is gradient descent, which modifies
# #each varaible according to the magnitude of the derivative of loss with respect to that variable.
# 
# optimizer = tf.train.GradientDescentOptimizer(0.01) #does .01 have to do with the step size
# train = optimizer.minimize(loss)
# # print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
# sess=tf.Session()
# #sess.run(init)
# 
# sess.run(init) # reset values to incorrect defaults.
# for i in range(1000):  
#     sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
#  #tested this, and it really does take this long.  maybe the .01 is a small step size
# print(sess.run([W, b]))


#now here is the complete program as presented by the makers of the tutorial.  




# Model parameters
# W = tf.Variable([.3], tf.float32)
# b = tf.Variable([-.3], tf.float32)
# # Model input and output
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b
# y = tf.placeholder(tf.float32)
# # loss
# loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# # optimizer
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
# # training data
# x_train = [1,2,3,4]
# y_train = [0,-1,-2,-3]
# # training loop
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init) # reset values to wrong
# for i in range(1000):
#     sess.run(train, {x:x_train, y:y_train})
# # evaluate training accuracy
# curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

#Next we come to the tf.contrib.learn library, which is supposed to simplify things.


#here we declare our features, including the type.  
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

#the estimator is the front end to do fitting and inference (logistic reg, linear reg)  This one is linear
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

#given some np arrays, we need to set up teh data sets using a helper function, which also needs how many batches (epochs) 
#we want and how big each batch should be
x=np.array([1.,2.,3.,4.])
y=np.array([0.,-1.,-2.,-3.])
input_fn= tf.contrib.learn.io.numpy_input_fn({"x":x},y,batch_size=4, num_epochs=1000)

#now we designate how many training steps we will take, I need to distinguish between epochs and training steps 
estimator.fit(input_fn=input_fn,steps=1000)
#and the evaluation is below
estimator.evaluate(input_fn=input_fn)
