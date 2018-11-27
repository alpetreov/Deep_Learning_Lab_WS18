import tensorflow as tf
import numpy as np

class Model:
    
    def __init__(self, lr, sess):
        
        # TODO: Define network
        height = 96
        width = 96

        # number of neurons in each layer
        input_num_units = 96*96
        output_num_units = 5

        # 1st conv layer
        conv1_nfilters = 32
        conv1_ksize = 5
        conv1_stride = 1
        conv1_pad = 'SAME'

        # 2nd conv layer
        conv2_nfilters = 64
        conv2_ksize = 5
        conv2_stride = 1
        conv2_pad = 'SAME'

        #parameters of fully connected network and outputs
        n_fc1 = 64
        n_outputs = 5
        flat_height = np.int32(height / 4) # division by 2 x number of max_pool layers
        print(flat_height)
        flat_width = np.int32(width / 4) # division by 2 x number of max_pool layers

        learning_rate = lr
        
        # define placeholders
        with tf.name_scope("inputs"):
            X = tf.placeholder(tf.float32, shape=[None, height, width], name = "X")
            X_reshaped = tf.reshape(X, shape=[-1,height,width,1])
            y = tf.placeholder(tf.int32, shape = [None, output_num_units], name = "y")

        conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_nfilters, kernel_size = conv1_ksize,
                                 strides = conv1_stride, padding=conv1_pad,
                                 activation = tf.nn.relu, name="conv1")

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(pool1, filters=conv2_nfilters, kernel_size=conv2_ksize,
                                 strides=conv2_stride, padding=conv2_pad,
                                 activation=tf.nn.relu, name="conv2")

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, shape=[-1,conv2_nfilters*flat_height*flat_width])

        fc1 = tf.layers.dense(pool2_flat, n_fc1, activation = tf.nn.relu,
                                  name = "fc1")

        with tf.name_scope("output"):
            logits = tf.layers.dense(fc1, n_outputs, name = "output") # logits a.k.a. y_hat
            Y_proba = tf.nn.softmax(logits, name="Y_proba")
        # TODO: Loss and optimizer
        with tf.name_scope("train"):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
            self.loss = tf.reduce_mean(xentropy)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            
        with tf.name_scope("eval"):
            pred_temp = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(pred_temp, tf.float32))
        

        # TODO: Start tensorflow session
        with tf.name_scope("init_and_save"):
            self.init = tf.global_variables_initializer()
        
        self.sess = sess;
        self.sess.run(init)
        self.saver = tf.train.Saver()

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
