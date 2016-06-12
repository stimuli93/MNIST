import tensorflow as tf
import numpy as np

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

class CNNTensorflow:
    def __init__(self):
        self.W_conv1 = weight_variable([3,3,1,8])
        self.b_conv1 = bias_variable([8])
        self.W_conv2 = weight_variable([3,3,8,16])
        self.b_conv2 = bias_variable([16])
        self.W_fc1 = weight_variable([7*7*16,512])
        self.b_fc1 = bias_variable([512])
        self.W_fc2 = weight_variable([512,10])
        self.b_fc2 = bias_variable([10])
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        self.X_inp = tf.placeholder(tf.float32, [None,784])
        self.y_inp = tf.placeholder(tf.float32, [None,10])
        self.keep_prob = tf.placeholder(tf.float32)
   
    def predict(self):
        x_image = tf.to_float(tf.reshape(self.X_inp,[-1,28,28,1]))

        h_conv1 = tf.nn.relu(conv2d(x_image,self.W_conv1) + self.b_conv1)
        h_pool1 = max_pool2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1,self.W_conv2) + self.b_conv2)
        h_pool2 = max_pool2x2(h_conv2)
        h_pool2_flattened = tf.reshape(h_pool2,[-1,7*7*16])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flattened,self.W_fc1) + self.b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1,self.keep_prob)
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,self.W_fc2) + self.b_fc2)
        return y_conv


    def predict_labels(self,X_test):
        y_conv = self.predict()
        y_matrix = self.sess.run(y_conv,feed_dict={self.X_inp:X_test})
        y_pred = np.argmax(y_matrix,axis=1)
        return y_pred    

    def train(self, X, y):    
        
        y_conv = self.predict()
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_inp*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)),reduction_indices=[1]))
        temp = set(tf.all_variables())
        train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)
        #Initializing only ADAM parameters
        self.sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

        correct_prediction = tf.equal(tf.argmax(self.y_inp,1),tf.argmax(y_conv,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        for i in range(1000):
            ids = np.random.choice(X.shape[0],200)
            x_batch = X[ids]
            y_batch = y[ids]
            if i%1 == 0:
                train_accuracy = self.sess.run(accuracy,feed_dict={self.X_inp:x_batch,self.y_inp:y_batch,self.keep_prob:0.5})
                print "Iteration %d, Train accuracy : %g"%(i,train_accuracy)
            self.sess.run(train_step,feed_dict={self.X_inp:x_batch,self.y_inp:y_batch,self.keep_prob:0.5})


