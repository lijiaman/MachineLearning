import os
import tensorflow as tf

import numpy as np
import time
import inspect

ALEX_MEAN = [103.939, 116.779, 123.68]


class ALEXNET:
    """
    An implementation of AlexNet model.
    """

    def __init__(self, alex_npy_path=None, trainable=True):
        if alex_npy_path is not None:
            self.data_dict = np.load(alex_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the NIN
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [227, 227, 1]
        assert green.get_shape().as_list()[1:] == [227, 227, 1]
        assert blue.get_shape().as_list()[1:] == [227, 227, 1]
        bgr = tf.concat(3, [
            blue - ALEX_MEAN[0],
            green - ALEX_MEAN[1],
            red - ALEX_MEAN[2],
        ])

        rgb = tf.concat(3, [
            red - ALEX_MEAN[0],
            green - ALEX_MEAN[1],
            blue - ALEX_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [227, 227, 3]

        self.conv1 = self.conv_layer(bgr, 3, 96, 11, 4, "conv1",padding="VALID")
        self.lrn1 = tf.nn.lrn(self.conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        #print("lrn1.shape:{0}".format(self.lrn1.get_shape()))
        self.pool1 = self.max_pool(self.lrn1, 3, 2, "pool1")
        #print "pool1.shape:"
        #print self.pool1.get_shape()
        self.conv2 = self.conv_group_layer(self.pool1, 96, 256, 5, 1, "conv2", 2)
        #print "conv2.shape:"
        #print self.conv2.get_shape()
        #self.conv2 = self.conv_layer(self.pool1, 96, 256, 5, 1, "conv2")
        self.lrn2 = tf.nn.lrn(self.conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        self.pool2 = self.max_pool(self.lrn2, 3, 2, "pool2")
        #print "pool2.shape:"
        #print self.pool2.get_shape()
        self.conv3 = self.conv_layer(self.pool2, 256, 384, 3, 1, "conv3")
        #print "conv3:"
        #print self.conv3.get_shape()
        self.conv4 = self.conv_group_layer(self.conv3, 384, 384, 3, 1, "conv4", 2)
        #self.conv4 = self.conv_layer(self.conv3, 384, 384, 3, 1, "conv4", [1,1])
        #self.conv5 = self.conv_layer(self.conv4, 384, 256, 3, 1, "conv5")
        self.conv5 = self.conv_group_layer(self.conv4, 384, 256, 3, 1, "conv5", 2)
        self.pool3 = self.max_pool(self.conv5, 3, 2, "pool3")
        #print "pool3.shape:"
        #print self.pool3.get_shape()
        self.fc6 = self.fc_layer(self.pool3, 9216, 4096, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, 0.5), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)
        
        self.fc7 = self.fc_layer(self.fc6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, 0.5), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)

        self.fc8 = self.fc_layer(self.relu7, 4096, 8, "fc8_final")

        self.data_dict = None

    def avg_pool(self, bottom, filter_size, stride, name):
        return tf.nn.avg_pool(bottom, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding='VALID', name=name)

    def max_pool(self, bottom, filter_size, stride, name):
        return tf.nn.max_pool(bottom, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding='VALID', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, filter_size, stride, name, padding="SAME", pad=None):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filter_size, in_channels, out_channels, name)
	    if pad is not None:
	        bottom = tf.pad(bottom, [[0,0],[pad[0],pad[0]],[pad[1],pad[1]],[0,0]],"CONSTANT")
            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            print name
            print relu.get_shape()
            return relu
    def conv_group_layer(self, bottom, in_channels, out_channels, filter_size, stride, name, group,pad=None):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filter_size, in_channels/group, out_channels, name)
            c_in = bottom.get_shape()[-1]
            assert c_in % group == 0
            assert out_channels % group == 0
            
            if pad is not None:
                bottom = tf.pad(bottom, [[0, 0],[pad[0],pad[0]],[pad[1],pad[1]],[0,0]], "CONSTANT") 
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, stride, stride, 1],padding='SAME')
            input_groups = tf.split(3, group, bottom)
            kernel_groups = tf.split(3, group, filt)
            output_groups = [convolve(i ,k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)
            return tf.reshape(tf.nn.bias_add(conv, conv_biases), [-1]+conv.get_shape().as_list()[1:])

    def cccp_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(1, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            print name
            print relu.get_shape()
            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            print name
            print fc.get_shape()
            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        #print "var.get_shape():"
        #print var.get_shape()
        #print "initial.get_shape()"
        #print initial_value.get_shape()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="/ais/gobi4/fashion/scene_alex.npy"):
        assert isinstance(sess, tf.InteractiveSession)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if not data_dict.has_key(name):
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
