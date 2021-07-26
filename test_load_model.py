# Save and restore 
import tensorflow as tf
import os

# Create variable 
a = tf.get_variable("A", initializer=tf.constant(3, shape=[2]))
b = tf.get_variable("B", initializer=tf.constant(5, shape=[3]))


# Create saver object 
saver = tf.train.Saver()
for i, var  in enumerate(saver._var_list):
    print('Var {}: {}'.format(i, var))

# initialize all of the variables
init_op = tf.global_variables_initializer()

# Run session 
with tf.Session() as sess:
    sess.run(init_op)
    # a_out, b_out = sess.run([a, b])
    # print('a = ', a_out)
    # print('b = ', b_out)

    # Save the variable in the disk
    save_path = saver.save(sess, './saved_variable')
    print('test_model saved in {}'.format(save_path))


# After that, check the file
for file in os.listdir('.'):
    if 'saved_variable' in file:
        print(file)

