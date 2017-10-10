from network import *
from module import *
from data import *
import argparse

class Graph:
    def __init__(self):
        self.graph = tf.Graph()

        with self.graph.as_default():

            data, num_batch = get_images(hp.batch_size)
            self.num_batch = num_batch
            x = data
            self.z = tf.placeholder(tf.float32, [hp.batch_size, 100], name="noise")

            gen = generator(self.z)
            fake_D = discriminator(gen)
            real_D = discriminator(x, reuse=True)

            if hp.loss_fn == 'standard':
                real_D, fake_D = tf.nn.sigmoid(real_D), tf.nn.sigmoid(fake_D)
                real_D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D, labels=tf.ones_like(real_D)))
                fake_D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D, labels=tf.zeros_like(fake_D)))
                self.disc_loss = (real_D_loss + fake_D_loss) / 2
                self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D, labels=tf.ones_like(fake_D))) / 2

            elif hp.loss_fn == 'ls':
                real_D, fake_D = tf.nn.sigmoid(real_D), tf.nn.sigmoid(fake_D)
                self.disc_loss= tf.reduce_mean(tf.square(1 - real_D) + tf.square(fake_D)) / 2
                self.gen_loss = tf.reduce_mean(tf.square(1 - fake_D)) / 2

            elif hp.loss_fn == 'w':
                self.disc_loss = -tf.reduce_mean(real_D) + tf.reduce_mean(fake_D)
                self.gen_loss = -tf.reduce_mean(fake_D)

            elif hp.loss_fn == "impw":
                lamb = 10.
                epsilon = np.random.uniform(0.0, 1.0, x.get_shape())
                x_hat = epsilon * x + (1.-epsilon) * gen
                x_hat_D = discriminator(x_hat, reuse=True)
                x_hat_gradient = tf.gradients(x_hat_D, [x_hat])[0]
                gradient_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(x_hat_gradient), reduction_indices=[1]))
                regularizer_loss = tf.reduce_mean(tf.square(gradient_l2_norm - 1.))

                self.disc_loss = -tf.reduce_mean(real_D) + tf.reduce_mean(fake_D) + lamb * regularizer_loss
                self.gen_loss = -tf.reduce_mean(fake_D)

            else: # "dra"
                lamb = 10.
                alpha = np.random.uniform(0.0, 1.0, x.get_shape())
                x_perturbed = x + 0.5 * reduce_std(x) * np.random.uniform(0.0,1.0,x.get_shape())
                x_hat = alpha * x + (1. - alpha) * x_perturbed
                x_hat_D = discriminator(x_hat, reuse=True)
                x_hat_gradient = tf.gradients(x_hat_D, [x_hat])[0]
                gradient_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(x_hat_gradient), reduction_indices=[1]))
                regularizer_loss = tf.reduce_mean(tf.square(gradient_l2_norm - 1.))

                real_D, fake_D = tf.nn.sigmoid(real_D), tf.nn.sigmoid(fake_D)
                real_D_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D, labels=tf.ones_like(real_D)))
                fake_D_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D, labels=tf.zeros_like(fake_D)))
                self.disc_loss = (real_D_loss + fake_D_loss) / 2 + lamb * regularizer_loss
                self.gen_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D, labels=tf.ones_like(fake_D))) / 2


            disc_op = tf.train.AdamOptimizer(learning_rate=0.0001)
            gen_op = tf.train.AdamOptimizer(learning_rate=0.0001)

            gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
            disc_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

            disc_grad = disc_op.compute_gradients(self.disc_loss, disc_variables)
            gen_grad = disc_op.compute_gradients(self.gen_loss, gen_variables)

            self.update_D = disc_op.apply_gradients(disc_grad)
            self.update_G = gen_op.apply_gradients(gen_grad)

            if hp.loss_fn == 'w':
                self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01))for var in disc_variables]

            tf.summary.scalar("generator_loss", self.gen_loss)
            tf.summary.scalar("discriminator_loss", self.disc_loss)

            tf.summary.image("real", x)
            tf.summary.image("generated", gen)

            self.merged = tf.summary.merge_all()

def main(args):

    g = Graph()
    global_step = 0

    with g.graph.as_default():

        sv = tf.train.Supervisor(logdir=hp.save_dir, save_model_secs=0, summary_op=None)

        with sv.managed_session() as sess:
            train_writer = tf.summary.FileWriter(hp.save_dir + '/train', sess.graph)

            if args.restore_step:
                restore_path = hp.save_dir + '/model_%d' % args.restore_step
                sv.saver.restore(sess, restore_path)
                global_step += args.restore_step
                print "Restore checkpoint %d succesfully!" % args.restore_step
            else:
                print "Start new training..."

            for epoch in xrange(hp.num_epochs):
                if sv.should_stop(): break

                for _ in range(g.num_batch):
                    z = np.random.uniform(-0.1, 0.1, [hp.batch_size, 100]).astype(np.float32)

                    if hp.loss_fn == 'w':
                        sess.run(g.update_D, feed_dict={g.z: z})
                        sess.run(g.clip_D, feed_dict={g.z: z})
                    elif hp.loss_fn == 'impw':
                        for _ in range(5):
                            sess.run(g.update_D, feed_dict={g.z: z})
                    else:
                        sess.run(g.update_D, feed_dict={g.z: z})
                    sess.run(g.update_G, feed_dict={g.z: z})

                    global_step += 1
                    if global_step % 100 == 0:
                        print global_step

                        d_loss, g_loss, summary = sess.run([g.disc_loss, g.gen_loss, g.merged], feed_dict={g.z:z})

                        print d_loss
                        print g_loss
                        train_writer.add_summary(summary, global_step)
                        sv.saver.save(sess, hp.save_dir + '/model_%d' % global_step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint')
    args = parser.parse_args()
    main(args)