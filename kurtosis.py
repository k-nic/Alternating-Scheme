#!/usr/bin/env python
# coding: utf-8

# In[1]:

#kurtosis is AS method for minimization of distance based on higher moments

import numpy as np
import tensorflow as tf
from keras_radam.training import RAdamOptimizer 


# In[2]:


def kurtosis (data_matrix, k=3, Lambda=10.0, Num_iter = 30, gamma = 1.0, BATCH_SIZE = 1000, number_of_moments = 1000, C=10.0):

    dimensionality = data_matrix.shape[1]
    number_of_points = data_matrix.shape[0]
    number_of_neurons = number_of_points
    Data = np.float32(np.transpose(data_matrix))
    
    max_grad_norm = 1000
    lr_decay = 0.3

    lambda1 = 1.0
    lambda2 = 1.0
    lambda3 = 1.0
    lambda4 = 1.0

    
    def model(inputs):
        with tf.variable_scope('Characteristic', reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable('w',initializer=tf.random_normal([number_of_neurons,1], stddev=0.0))
            B = tf.get_variable('B',initializer=tf.convert_to_tensor(Data))
        out = (2/number_of_neurons)*tf.matmul(tf.math.cos(tf.matmul(inputs, B)), tf.nn.sigmoid(w))
        return out
    
    def gradients(inputs):
        with tf.variable_scope('Characteristic', reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable('w',initializer=tf.random_normal([number_of_neurons,1], stddev=0.0))
            B = tf.get_variable('B',initializer=tf.convert_to_tensor(Data))
        out = (2/number_of_neurons)*tf.matmul(tf.math.cos(tf.matmul(inputs, B)), tf.nn.sigmoid(w))
        grads = tf.reshape(tf.gradients(out, [inputs])[0], [BATCH_SIZE, dimensionality])
        return grads
    
    def old_gradients(inputs, old_weights):
        old_B, old_w = old_weights
        old = (2/number_of_neurons)*tf.matmul(tf.math.cos(tf.matmul(inputs, old_B)), tf.nn.sigmoid(old_w))
        return tf.reshape(tf.gradients(old, [inputs])[0], [BATCH_SIZE, dimensionality])
    
    new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    Dataplace = tf.placeholder(tf.float32, shape=(dimensionality, None))
    
    tf_data_x = 0.0001*tf.random_normal([BATCH_SIZE, dimensionality]) #tf.placeholder(tf.float32, shape=(1, dimensionality)) #tf.zeros(shape=(1, dimensionality), dtype=tf.dtypes.float32) # 
    tf_data_y = tf.reduce_mean(tf.math.cos(tf.matmul(tf_data_x, Dataplace))) #
    
    tf_data_w = tf.placeholder(tf.float32, shape=(number_of_neurons,1))
    tf_data_B = tf.placeholder(tf.float32, shape=(dimensionality,number_of_neurons))
    tf_data_P = tf.placeholder(tf.float32, shape=(dimensionality, dimensionality))
    
    tf_data_second = gamma*tf.random_normal([BATCH_SIZE, dimensionality])
    Lbd = tf.placeholder(tf.float32, shape=[], name="lambda")
    
    prediction = model(tf_data_x)
    with tf.variable_scope('Characteristic', reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable('w')
    penalty = tf.square((2/number_of_neurons)*tf.reduce_sum(tf.nn.sigmoid(w))-1.0)
    
    rand_index1 = tf.random.uniform(shape=[number_of_moments,1], minval=0, maxval=dimensionality, dtype=tf.int32, seed=10)
    rand_index2 = tf.random.uniform(shape=[number_of_moments,2], minval=0, maxval=dimensionality, dtype=tf.int32, seed=10)
    rand_index3 = tf.random.uniform(shape=[number_of_moments,3], minval=0, maxval=dimensionality, dtype=tf.int32, seed=10)
    rand_index4 = tf.random.uniform(shape=[number_of_moments,4], minval=0, maxval=dimensionality, dtype=tf.int32, seed=10)
    
    phi = prediction - tf_data_y
    collect = 0.0
    for i in range(0,number_of_moments):
        a = rand_index1[i][0]
        b = rand_index2[i]
        c = rand_index3[i]
        d = rand_index4[i]
        r = tf.gradients(phi, [tf_data_x[0,a]], unconnected_gradients='zero')[0]
        collect = collect + lambda1*r*r
        r = tf.gradients(phi, [tf_data_x[0,b[0]]], unconnected_gradients='zero')[0]
        r = tf.gradients(r, [tf_data_x[0,b[1]]], unconnected_gradients='zero')[0]
        collect = collect + lambda2*r*r
        r = tf.gradients(phi, [tf_data_x[0,c[0]]], unconnected_gradients='zero')[0]
        r = tf.gradients(r, [tf_data_x[0,c[1]]], unconnected_gradients='zero')[0]
        r = tf.gradients(r, [tf_data_x[0,c[2]]], unconnected_gradients='zero')[0]
        collect = collect + lambda3*r*r
        r = tf.gradients(phi, [tf_data_x[0,d[0]]], unconnected_gradients='zero')[0]
        r = tf.gradients(r, [tf_data_x[0,d[1]]], unconnected_gradients='zero')[0]
        r = tf.gradients(r, [tf_data_x[0,d[2]]], unconnected_gradients='zero')[0]
        r = tf.gradients(r, [tf_data_x[0,d[3]]], unconnected_gradients='zero')[0]
        collect = collect + lambda4*r*r
    out_loss = collect/number_of_moments + C*penalty
    
    new_part = gradients(tf_data_second)
    
    tf_data_grad_psi = old_gradients(tf_data_second, [tf_data_B, tf_data_w])
    
    old_part = tf.matmul(tf_data_grad_psi, tf_data_P)
    
    loss = out_loss + Lbd*tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(new_part, old_part)), axis=1))
    
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    optimizer = RAdamOptimizer(learning_rate=new_lr, beta1=0.5, beta2=0.9)
    target = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
    
    cur_w = np.random.normal(0, 0.35, (number_of_neurons,1))
    cur_B =  np.random.normal(0, 0.35, (dimensionality, number_of_neurons))
    cur_P = np.zeros((dimensionality, dimensionality))
    O = np.zeros((dimensionality, k))
    
    lr = 0.0001
    prev_run_res = 1000000000
    
    # default session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for s in range(0, Num_iter):
    
        lr = 0.001
        epoch = 0
        while lr > 0.0000001:  # iterations epoch < 500000 and 
            
            sess.run(target, feed_dict={new_lr: lr, Dataplace: Data, Lbd: Lambda,
                tf_data_w: cur_w,
                tf_data_B: cur_B,
                tf_data_P: cur_P,
            })
    
            if epoch % 100 == 0:
                run_res = 0.0
                for times in range(100):
                    run_res = run_res + sess.run(loss, feed_dict={Dataplace: Data, Lbd: Lambda, 
                                                   tf_data_w: cur_w,
                                                   tf_data_B: cur_B,
                                                   tf_data_P: cur_P})
                run_res = run_res/100.0
                print("epoch = %d, train error = %.8f" % (epoch + 1, run_res))
    
                if  run_res > prev_run_res:
                    lr *= lr_decay
                prev_run_res = run_res
    
            epoch += 1
            
        with tf.variable_scope('Characteristic', reuse=tf.AUTO_REUSE) as scope:
            w1 = tf.get_variable('w')
            B1 = tf.get_variable('B')
        
        cur_w, cur_B = sess.run([w1, B1])    
    
        third_grad_psi = np.reshape(sess.run([new_part]), (BATCH_SIZE, dimensionality))
        for r in range(1000):
            sess.run([tf_data_second])
            np.concatenate((third_grad_psi, np.reshape(sess.run([new_part]), (BATCH_SIZE, dimensionality))), axis=0) 
    
        u, s, vh = np.linalg.svd(np.matmul(np.transpose(third_grad_psi), third_grad_psi), full_matrices=True)
        O = u[:,0:k:1]
    #    print(s)
    #    print(u)
    #    print(O)
        cur_P = np.matmul(O, np.transpose(O))
    #    print(cur_P)
    
        
        print(np.sum(np.square(third_grad_psi)))
        print(np.sum(np.square(third_grad_psi-np.matmul(third_grad_psi, cur_P))))
    
    sess.close()
    return O

