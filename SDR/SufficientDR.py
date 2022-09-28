#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
from scipy.stats import logistic

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 

# In[2]:


def sigmoid(x):
  return 1. /(1+np.exp(-x))

def sdr(train_x, train_y, k=2, gamma = 1.0, epsilon = 0.8, Lambda = 10000.0,         number_of_neurons = 200, BATCH_SIZE = 45, num_epochs = 50, num_iters = 1000, classify = False):
    dimensionality = train_x.shape[1]
    num_chunks = int(train_x.shape[0]/BATCH_SIZE)
    if (num_chunks*BATCH_SIZE != train_x.shape[0]):
        print("BATCH_SIZE should be chosen as a divisor of the number of examples to use the whole train set\n")
    max_grad_norm = 1000
    
    BATCH_SIZE_2 = 100
    M = 30000
    
    new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
     
    tf_data_x = tf.placeholder(tf.float32, shape=(None,dimensionality)) 
    tf_data_y = tf.placeholder(tf.float32, shape=(None,1)) 
    tf_data_w = tf.placeholder(tf.float32, shape=(number_of_neurons,1))
    tf_data_B = tf.placeholder(tf.float32, shape=(dimensionality, number_of_neurons))
    tf_data_biases = tf.placeholder(tf.float32, shape=(number_of_neurons))
    tf_data_P = tf.placeholder(tf.float32, shape=(dimensionality, dimensionality))
    
    tf_data_second = gamma*tf.random_normal([BATCH_SIZE_2, dimensionality])
    Lbd = tf.placeholder(tf.float32, shape=[], name="lambda")
    
    w = tf.Variable(tf.random_normal([number_of_neurons,1], stddev=0.35), name="neuron_weights")
    B = tf.Variable(tf.random_normal([dimensionality, number_of_neurons], stddev=0.35), name="weights")
    biases = tf.Variable(tf.zeros([number_of_neurons]), name="biases")
    
    noise = epsilon*tf.random_normal([BATCH_SIZE,dimensionality])
    if classify:
        logits = tf.matmul(tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(tf_data_x+noise, B), biases)), w)
        out_loss = -tf.reduce_mean(tf.multiply(logits, tf_data_y)) + tf.reduce_mean(tf.math.log(1.0+tf.math.exp(logits)))
    else:
        prediction = tf.matmul(tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(tf_data_x+noise, B), biases)), w)
        out_loss = tf.reduce_mean(tf.square(prediction - tf_data_y))
    
    sigma_prime = tf.multiply(1.0 - tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(tf_data_second, B), biases)), 
                           tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(tf_data_second, B), biases)))
    multiply = tf.stack([BATCH_SIZE_2, 1]) 
    w_M_times = tf.tile(tf.transpose(w), multiply)
    grad_psi = tf.matmul(tf.multiply(w_M_times, sigma_prime), tf.transpose(B))
    
    new_part = grad_psi
    
    tf_data_sigma_prime = tf.multiply(1.0 - tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(tf_data_second, tf_data_B), tf_data_biases)), 
                           tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(tf_data_second, tf_data_B), tf_data_biases)))
    tf_data_w_M_times = tf.tile(tf.transpose(tf_data_w), multiply)
    tf_data_grad_psi = tf.matmul(tf.multiply(tf_data_w_M_times, tf_data_sigma_prime), tf.transpose(tf_data_B))
    
    old_part = tf.matmul(tf_data_grad_psi, tf_data_P)
    
    loss = out_loss + Lbd*tf.reduce_mean(tf.square(tf.subtract(new_part, old_part)))
    
    target = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(loss)

    cur_w = np.random.normal(0, 0.35, (number_of_neurons,1))
    cur_B =  np.random.normal(0, 0.35, (dimensionality, number_of_neurons))
    cur_biases = np.zeros((number_of_neurons))
    cur_P = np.zeros((dimensionality, dimensionality)) 
    O = np.zeros((dimensionality, k)) 
    cur_iter = 0
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for epoch in range(num_epochs):
#        print("Epoch %d" %(epoch))
        for iteration in range(num_iters):
            offset = (cur_iter % num_chunks)*BATCH_SIZE
            sample_x = np.reshape(train_x[offset:(offset+BATCH_SIZE)], (BATCH_SIZE, dimensionality))
            sample_y = np.reshape(train_y[offset:(offset+BATCH_SIZE)], (BATCH_SIZE, 1))
            sess.run(target, feed_dict={tf_data_x: sample_x, tf_data_y: sample_y, 
                                  tf_data_w: cur_w, tf_data_B: cur_B, tf_data_biases: cur_biases,
                                  tf_data_P: cur_P, Lbd: Lambda, batch_size: BATCH_SIZE})
            cur_iter = cur_iter+1
        reses = []
        outes = []
        for i in range(num_chunks):
            offset = (cur_iter % num_chunks)*BATCH_SIZE
            sample_x = np.reshape(train_x[offset:(offset+BATCH_SIZE)], (BATCH_SIZE, dimensionality))
            sample_y = np.reshape(train_y[offset:(offset+BATCH_SIZE)], (BATCH_SIZE, 1))
            res, out = sess.run([loss, out_loss], feed_dict={tf_data_x: sample_x, tf_data_y: sample_y, 
                                  tf_data_w: cur_w, tf_data_B: cur_B, tf_data_biases: cur_biases,
                                  tf_data_P: cur_P, Lbd: Lambda, batch_size: BATCH_SIZE})
            reses.append(res)
            outes.append(out)
            cur_iter = cur_iter+1
#        print ("Iter %d: loss: %.4f square loss part: %.4f\n" %(cur_iter, np.mean(np.array(reses)), np.mean(np.array(outes))))

        cur_w, cur_B, cur_biases = sess.run([w, B, biases], feed_dict={})    
        third_x = np.random.normal(0, gamma, (M,dimensionality))
        third_sigma_prime = np.multiply(1.0-logistic.cdf(np.add(np.matmul(third_x, cur_B), np.tile(np.reshape(cur_biases,[1, number_of_neurons]),(M,1)))), 
                           logistic.cdf(np.add(np.matmul(third_x, cur_B), np.tile(np.reshape(cur_biases,[1, number_of_neurons]),(M,1)))))
        third_w_M_times = np.tile(np.reshape(cur_w,[1, number_of_neurons]), (M,1))
        third_grad_psi = np.matmul(np.multiply(third_w_M_times, third_sigma_prime), np.transpose(cur_B))
        
        u, s, vh = np.linalg.svd(np.transpose(third_grad_psi), full_matrices=True)
        O = u[:,0:k:1]
#        print(s)
        cur_P = np.matmul(O, np.transpose(O))
        tvr = 1-np.sum(np.multiply(s[0:k],s[0:k]))/np.sum(np.multiply(s,s))
    sess.close()
    return O

