import tensorflow as tf


tf.reset_default_graph()

imported_graph = tf.train.import_meta_graph('./models_0/orig-pre_it800.ckpt.meta')

# list all tensors in the graph
i = -1
for tensor in tf.get_default_graph().get_operations():
    i += 1
    if i <= 50:
        print(tensor.name)

print(i + 1)
with tf.device("/job:localhost/replica:0/task:0/device:CPU:0"):
    with tf.Session() as sess:
        imported_graph.restore(sess, './models_0/orig-pre_it800.ckpt')
        a = sess.run(['Placeholder:0'])
        print(a)