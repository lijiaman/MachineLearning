import tensorflow as tf
import os
import numpy as np
import time
import inspect
import skimage
import skimage.io
import skimage.transform
import input
import native_alex

sess = tf.InteractiveSession()
batch_size = 64
x = tf.placeholder("float", [None, 227, 227, 3])
y_ = tf.placeholder(tf.int32, shape=[None,])
train_mode = tf.placeholder(tf.bool)

npy_path = '/ais/gobi4/fashion/bvlc_alexnet.npy'
network = native_alex.ALEXNET(alex_npy_path=npy_path, trainable=True)
network.build(x, train_mode)
y = network.fc8
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))

prediction = tf.cast(tf.argmax(y, 1), tf.int32)
correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.scalar_summary('Loss', cost)
tf.scalar_summary('Acc', acc)
merged = tf.merge_all_summaries()
log_dir = 'csc411_finetune'
train_writer = tf.train.SummaryWriter(log_dir+'/train')
val_writer = tf.train.SummaryWriter(log_dir+'/val')
# train
n_epoch = 500
global_step = tf.Variable(0)
starter_learning_rate = 0.0001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 320, 0.96, staircase=True)
print_freq = 1

train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost, global_step=global_step)
#train_op = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost,global_step=global_step)

sess.run(tf.initialize_all_variables())


ISOTIMEFORMAT = '%Y-%m-%d %X'

x_train_val, t_train_val = input.load_train_val_dataset()
num_train_cases = 4500
num_val_cases = 7000 - num_train_cases
x_train = x_train_val[:num_train_cases, :, :, :]
x_val = x_train_val[num_train_cases:-1, :, :, :]
t_train = t_train_val[:num_train_cases]
t_val = t_train_val[num_train_cases:-1]

rnd_idx = np.arange(x_train.shape[0])#From A2 

id_test, x_test = input.load_test_dataset()
num_test_cases = id_test.shape[0]
t_empty_test = np.zeros(num_test_cases)

num_steps = int(np.ceil(num_train_cases / batch_size))
num_val_steps = int(np.ceil(num_val_cases / batch_size))
num_test_steps = int(np.ceil(num_test_cases / batch_size))

iter_cnt = 0
for epoch in range(n_epoch):
    start_time = time.time()
    np.random.shuffle(rnd_idx)
    x_train = x_train[rnd_idx]
    t_train = t_train[rnd_idx]
    for step in xrange(num_steps):
        iter_cnt += 1
        start = step * batch_size
        end = min(num_train_cases, (step + 1) * batch_size)
        x_batch = x_train[start: end]
        t_batch = t_train[start: end]
        feed_dict = {x: x_batch, y_: t_batch, train_mode:True}
        #conv1, conv2, conv3, conv4, conv5, fc8, fc7, fc6, pool3 = sess.run([network.conv1, network.conv2, network.conv3, network.conv4, network.conv5, network.fc8, network.fc7, network.fc6, network.pool3], feed_dict=feed_dict)
        _, err, ac, lr, train_summary = sess.run([train_op, cost, acc, learning_rate, merged], feed_dict=feed_dict)
        train_writer.add_summary(train_summary, iter_cnt)
        #network.save_npy(sess=sess, npy_path="test_save.npy")

        if step % 5 == 0:
            print_time = time.strftime(ISOTIMEFORMAT, time.localtime())
            print("{0} Epoch: {1}, Train iteration: {2}, lr: {3}".format(print_time, epoch+1, step+1, lr))
            print("   Train Loss: %f" % err)
            print("   Top1 Acc: %f" % ac)
    
    val_sum_err = 0
    val_sum_acc = 0
    for val_step in xrange(num_val_steps):
        val_start = val_step * batch_size
        val_end = min(num_val_cases, (val_step + 1) * batch_size)
        x_val_batch = x_val[val_start: val_end]
        t_val_batch = t_val[val_start: val_end]
        feed_dict_val = {x: x_val_batch, y_: t_val_batch, train_mode:False}
        val_err, val_ac = sess.run([cost, acc], feed_dict=feed_dict_val)
        val_sum_err += val_err
        val_sum_acc += val_ac
           
    print_time = time.strftime(ISOTIMEFORMAT, time.localtime())
    print("{0} Epoch: {1}".format(print_time, epoch+1))
    print("   Val Loss: %f" % (val_sum_err/num_val_steps))
    print("   Val Top1 Acc: %f" % (val_sum_acc/num_val_steps))

    summary = tf.Summary()
    summary.value.add(tag='Acc', simple_value=val_sum_acc/num_val_steps)
    val_writer.add_summary(summary, iter_cnt)

    summary = tf.Summary()
    summary.value.add(tag='Loss', simple_value=val_sum_err/num_val_steps)
    val_writer.add_summary(summary, iter_cnt)
          


    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        feed_dict_test = {x: x_test, y_: t_empty_test, train_mode:False}
        pred = np.asarray(sess.run([prediction], feed_dict=feed_dict_test))
        pred = pred.reshape(-1)
        if os.path.isfile('prediction_result.txt'):
            os.remove('prediction_result.txt')
        #print("pred.shape:",np.asarray(pred).shape)
        #print("id_test.shape:", np.asarray(id_test).shape)
        with open('prediction_result.txt', 'a') as w_f:
            for i in xrange(id_test.shape[0]):
                w_f.write(str(id_test[i])+'\t'+str(pred[i]+1)+'\n')
        w_f.close()
        network.save_npy(sess=sess, npy_path="/ais/gobi4/fashion/scene_alex.npy")

train_writer.close()
val_writer.close()
