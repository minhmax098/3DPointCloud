import tensorflow as tf

tf.reset_default_graph()

# Import the graph from the file
imported_graph = tf.train.import_meta_graph('saved_variable.meta')
for tensor in tf.get_default_graph().get_operations():
    print(tensor.name)

# Run the session and read the value from these name
with tf.Session() as sess: 
    # Restore the saved saved_variable 
    imported_graph.restore(sess, './saved_variable')
    # print the loaded variable
    a_out, b_out = sess.run(['A:0', 'B:0'])
    print('a = ', a_out)
    print('b = ', b_out)

# Ok, nice :v 
# Then performing load variables from our saved data in model0 folder
