from module import *

def generator(tensor, scope="generator"):

    with tf.variable_scope(scope):
        if hp.loss_fn == 'impw' or hp.loss_fn == 'dra':
            tensor = fully_connected(tensor, 16 * 16 * 128, norm_fn=None, scope="fc1")
            tensor = tf.reshape(tensor, [-1, 16, 16, 128])
            tensor = conv2d(tensor, output_dim=4 * 64, norm_fn=None, scope="conv1")
            tensor = tf.reshape(tensor, [-1, 32, 32, 64])
            tensor = conv2d(tensor, output_dim=4 * 32, norm_fn=None, scope="conv2")
            tensor = tf.reshape(tensor, [-1, 64, 64, 32])
            tensor = conv2d(tensor, output_dim=3, norm_fn=None, activation_fn=tf.nn.tanh, scope="conv3")
        else:
            tensor = fully_connected(tensor, 16*16*128, scope="fc1")
            tensor = tf.reshape(tensor, [-1, 16, 16, 128])
            tensor = conv2d(tensor, output_dim=4*64, scope="conv1")
            tensor = tf.reshape(tensor, [-1,32,32,64])
            tensor = conv2d(tensor, output_dim=4*32, scope="conv2")
            tensor = tf.reshape(tensor, [-1,64,64,32])
            tensor = conv2d(tensor, output_dim=3, norm_fn=None, activation_fn=tf.nn.tanh, scope="conv3")

        return tensor


def discriminator(tensor, reuse=False, scope="discriminator"):
    with tf.variable_scope(scope, reuse=reuse):
        if hp.loss_fn == 'impw' or hp.loss_fn == 'dra':
            tensor = conv2d(tensor, stride=2, activation_fn=lrelu, norm_fn=None, output_dim=32, scope="conv1")
            tensor = conv2d(tensor, stride=2, activation_fn=lrelu, norm_fn=None, output_dim=64, scope="conv2")
            tensor = tf.reshape(tensor, [-1, 16 * 16 * 64])
            tensor = fully_connected(tensor, 512, norm_fn=None, scope="fc1")
            tensor = fully_connected(tensor, 1, norm_fn=None, scope="fc2", is_last=True)
        else:
            tensor = conv2d(tensor, stride=2, activation_fn=lrelu, output_dim=32, scope="conv1")
            tensor = conv2d(tensor, stride=2, activation_fn=lrelu, output_dim=64, scope="conv2")
            tensor = tf.reshape(tensor, [-1, 16*16*64])
            tensor = fully_connected(tensor, 512, scope="fc1")
            tensor = fully_connected(tensor, 1, scope="fc2", is_last=True)

        return tensor

# def network(x,z, scope="network"):
#     with tf.variable_scope(scope):
#         fake_x = generator(z)
#         disc_x = discriminator(x)
#         disc_z = discriminator(fake_x, reuse=True)
#
#         return fake_x, disc_x, disc_z

