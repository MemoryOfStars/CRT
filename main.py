#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:50:46 2018
Prediction on CTR
@author: kmkgmx
"""

import tensorflow as tf
import time



#features of train_set
integer_features = tf.placeholder('float',[1,13])
categorical_features = tf.placeholder('float',[1,26])
result = tf.placeholder('float')

x1 = tf.reshape(integer_features,[-1,1,13,1])
x2 = tf.reshape(categorical_features,[-1,1,26,1])
#1st Convolution Layer
filter1_int = tf.Variable(tf.truncated_normal([1, 5, 1, 6]))
filter1_cate = tf.Variable(tf.truncated_normal([1, 5, 1, 6]))
bias1_int = tf.Variable(tf.truncated_normal([6]))
bias1_cate = tf.Variable(tf.truncated_normal([6]))
conv1_int = tf.nn.conv2d(x1,filter1_int,strides=[1,1,1,1],padding='SAME')
conv1_cate = tf.nn.conv2d(x2,filter1_cate,strides=[1,1,1,1],padding = 'SAME')
h_conv1_int = tf.nn.relu(conv1_int + bias1_int)
h_conv1_cate = tf.nn.relu(conv1_cate + bias1_cate)

#Pooling Layer
##int_fea --->[1,12]
avgPool2_int = tf.nn.avg_pool(h_conv1_int,ksize=[1,1,2,1],strides=[1,1,1,1],padding='SAME')
##cate --->[1,13]
avgPool2_cate = tf.nn.avg_pool(h_conv1_cate,ksize=[1,1,2,1],strides=[1,2,1,1],padding='SAME')


#2nd Convolution Layer
filter2_int = tf.Variable(tf.truncated_normal([1,5,6,16]))
filter2_cate = tf.Variable(tf.truncated_normal([1,5,6,16]))
bias2_int = tf.Variable(tf.truncated_normal([16]))
bias2_cate = tf.Variable(tf.truncated_normal([16]))
conv2_int = tf.nn.conv2d(avgPool2_int,filter2_int,strides=[1,1,1,1],padding='SAME')
conv2_cate = tf.nn.conv2d(avgPool2_cate,filter2_cate,strides=[1,1,1,1],padding='SAME')
h_conv2_int = tf.nn.relu(conv2_int + bias2_int)
h_conv2_cate = tf.nn.relu(conv2_cate + bias2_cate)


#Pooling Layer
##int_fea--->[1,7]
avgPool3_int = tf.nn.avg_pool(h_conv2_int,ksize=[1,1,5,1],strides=[1,1,1,1],padding='SAME')
##cate_fea--->[1,7]
avgPool3_cate = tf.nn.avg_pool(h_conv2_cate,ksize=[1,1,6,1],strides=[1,1,1,1],padding='SAME')


#3rdConvolution Layer
filter3_int = tf.Variable(tf.truncated_normal([1,5,16,120]))
filter3_cate = tf.Variable(tf.truncated_normal([1,5,16,120]))
bias3_int = tf.Variable(tf.truncated_normal([120]))
bias3_cate = tf.Variable(tf.truncated_normal([120]))
conv3_int = tf.nn.conv2d(avgPool3_int,filter3_int,strides=[1,1,1,1],padding='SAME')
conv3_cate = tf.nn.conv2d(avgPool3_cate,filter3_cate,strides=[1,1,1,1],padding='SAME')
h_conv3_int = tf.nn.relu(conv3_int + bias3_int)
h_conv3_cate = tf.nn.relu(conv3_cate + bias3_cate)



W_fc1_int = tf.Variable(tf.truncated_normal([1 * 13 * 120,80])) 
W_fc1_cate = tf.Variable(tf.truncated_normal([1 * 13 * 120,80])) 

b_fc1_int = tf.Variable(tf.truncated_normal([80]))
b_fc1_cate = tf.Variable(tf.truncated_normal([80]))

h_pool2_flat_int = tf.reshape(h_conv3_int,[-1,1 * 13 * 120])
h_pool2_flat_cate = tf.reshape(h_conv3_cate,[-1,1 * 13 * 120])

h_fc1_int = tf.nn.relu(tf.matmul(h_pool2_flat_int,W_fc1_int) + b_fc1_int)
h_fc1_cate = tf.nn.relu(tf.matmul(h_pool2_flat_cate,W_fc1_cate) + b_fc1_cate)

#Output Layer
W_fc2_int = tf.Variable(tf.truncated_normal([80,1]))
W_fc2_cate = tf.Variable(tf.truncated_normal([80,1]))
b_fc2_int = tf.Variable(tf.truncated_normal([1]))
b_fc2_cate = tf.Variable(tf.truncated_normal([1]))
y_conv_int = tf.nn.softmax(tf.matmul(h_fc1_int,W_fc2_int) + b_fc2_int)
y_conv_cate = tf.nn.softmax(tf.matmul(h_fc1_cate,W_fc2_cate) + b_fc2_cate)

W_fc3_int = tf.Variable(tf.truncated_normal([1,1]))
W_fc3_cate = tf.Variable(tf.truncated_normal([1,1]))
b_fc3 = tf.Variable(tf.truncated_normal([1]))
y_conv = tf.nn.softmax(tf.matmul(y_conv_cate,W_fc3_cate) 
                        + tf.matmul(y_conv_int,W_fc3_int) 
                        + b_fc3)
#Loss
cross_entropy = -tf.reduce_sum(result * tf.log(y_conv))

train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

sess = tf.InteractiveSession()
#Evaluate Accuracy
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(result,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
#Initialize

tf.global_variables_initializer().run()


'''
train_part
'''
#train_op = 
#Get TrainSets
start_time = time.time()
with open('train.set.csv') as train_set:
    for line in train_set.readlines():
        #Transform into Matched Format
        #print(line.strip().split(",")[1])
        #print(line)
        '''
        col= tf.decode_csv(line, record_defaults=[[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],
                                                  [1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],
                                                  [1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],
                                                  [1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]], 
                           field_delim=',',name=None)
        batch_xs = col[2:]
        batch_ys = col[1:2]
        '''
        batch_xs = line.strip().split(",")[2:]
        batch_ys = line.strip().split(",")[1:2]
        print(batch_xs[11])

        i=0
        while i < len(batch_xs):
            if batch_xs[i] == '':
                batch_xs[i] = 0
            i = i + 1
        while i < 13:
            batch_xs[i] = int(batch_xs[i])
            i = i + 1
        i=13
        while i < len(batch_xs):
            if batch_xs[i] != 0:
                string = batch_xs[i]
                batch_xs[i] = int(string,16)
            i = i + 1
            
        i=0    
        while i < len(batch_ys):
            if batch_ys == '':
                batch_ys[i] = 0
            i = i + 1
        
        i=0
        while i < len(batch_xs):
            batch_xs[i] = float(batch_xs[i])
            i = i + 1
            
            
        i=0    
        while i < len(batch_ys):
            batch_ys[i] = float(batch_ys[i])
            i = i + 1
        
        print(batch_ys)
        #batch_y = float(batch_ys)
        #print(batch_y)
        print([1,batch_xs[:13]])
        train_accuracy = accuracy.eval(feed_dict={integer_features:[1,batch_xs[:13]],
                                                  categorical_features:[1,batch_xs[13:]],
                                                  result: batch_ys[0]})
        print("step %d,training accuracy %g" % (line[0],train_accuracy))
        #Time Between
        end_time = time.time()
        print('time:',(end_time - start_time))
        start_time = end_time
        #Train Started
        train_step.run(feed_dict={integer_features:[1,batch_xs[:13]],
                                  categorical_features:[1,batch_xs[13:]],
                                  result: batch_ys[0]})
    
    #Close Session
    sess.close()
