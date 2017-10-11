#for deep learning course in NCTU 2017
#if you have questions, mail to followwar@gmail.com


from __future__ import print_function
import tensorflow as tf

#use others' data loader
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#define placeholder
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1, 28, 28, 1]) # resize back to 28 x 28


## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)     # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

## dropout ##
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_fc = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


## the learning rate  
# learning rate 0.1 for first 80 epoch (469 iteration for 1 epoch)
# decay learning rate to 0.01 at 81th epoch
# decay learning rate to 0.01 at 121th epoch
global_step = tf.Variable(0, trainable=False)
boundaries = [37520, 56280]
values = [0.1, 0.01, 0.001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)



# the loss function 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_fc))

# optimizer SGD
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# prediction
correct_prediction = tf.equal(tf.argmax(y_fc,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start session
sess = tf.InteractiveSession()
# initize variable 
sess.run(tf.global_variables_initializer())

for i in range(76916): # 164 epoch for all (496*164)
  batch = mnist.train.next_batch(128)
  ## test every 100 step
  if i%100 == 0:
    print("step %d, Test accuracy %g"%(i, (accuracy.eval(feed_dict={
        x:mnist.test.images,y_: mnist.test.labels, keep_prob: 1.0}))))
  ## trianing
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

##final testing
print("Final test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
