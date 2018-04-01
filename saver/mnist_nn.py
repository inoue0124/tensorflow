#-*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("data/",one_hot=True)

# 入力データ
x = tf.placeholder(tf.float32,[None,784])

# 入力画像をログに出力
img = tf.reshape(x,[-1,28,28,1])
tf.summary.image("input_data",img,10)

# 入力層から中間層
with tf.name_scope("hidden"):
    w_1 = tf.Variable(tf.truncated_normal([784,64],stddev=0.1), name="w1")
    b_1 = tf.Variable(tf.zeros([64]), name="b1")
    h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)
    tf.summary.histogram('w_1',w_1)

# 中間層から出力層
with tf.name_scope("output"):
    w_2 = tf.Variable(tf.truncated_normal([64,10],stddev=0.1), name="w2")
    b_2 = tf.Variable(tf.zeros([10]), name="b2")
    out = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2)
    tf.summary.histogram('w_2',w_2)

# 誤差関数
with tf.name_scope("loss"):
    y = tf.placeholder(tf.float32, [None,10])
    loss = tf.reduce_mean(tf.square(y - out))
    tf.summary.scalar("loss",loss)

# 訓練
with tf.name_scope("train"):
    global_step = tf.Variable(0,name='global_step',trainable=False)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss,global_step=global_step)

# 評価
with tf.name_scope("accuracy"):
    correct = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar("accuracy",accuracy)

# 初期化
init = tf.global_variables_initializer()
summary_op = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep = 3)

with tf.Session() as sess:
    ckpt_state = tf.train.get_checkpoint_state('ckpt/')
    if ckpt_state:
        last_model = ckpt_state.model_checkpoint_path
        saver.restore(sess,last_model)
        print('model was loaded:',last_model)
    else:
        sess.run(init)
        print('initialized.')
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    summary_writer = tf.summary.FileWriter("logs",sess.graph)    
    last_step = sess.run(global_step)

    for i in range(1000):
        step = i + last_step
        train_images,train_labels = mnist.train.next_batch(50)
        sess.run(train_step,feed_dict={x:train_images,y:train_labels})

        if step % 10 == 0:
            acc_val = sess.run(accuracy,feed_dict={x:test_images,y:test_labels})
            summary_str = sess.run(summary_op,feed_dict={x:test_images,y:test_labels})
            print('Step %d: accuracy = %.2f' % (step, acc_val))
            summary_writer.add_summary(summary_str,step)
            saver.save(sess,'ckpt/my_model',global_step=step+1,write_meta_graph=False)
