import scipy.misc
import numpy as np

from os import listdir
from os.path import isfile, join
import tensorflow as tf

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  # h, w = x.get_shape().as_list()[:2]
  h, w = 218, 178
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return tf.image.resize_image_with_crop_or_pad(
      x[j:j+crop_h, i:i+crop_w], resize_h, resize_w)


def get_images(batch_size):

    min_after_dequeue = 32 * batch_size
    capacity = 64 * batch_size
    path = './data'
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    tf.convert_to_tensor(files, dtype=tf.string)
    num_batch = len(files) // batch_size

    filename_queue = tf.train.string_input_producer(files, shuffle=True)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    images = tf.image.decode_jpeg(value, channels=3)
    images = tf.image.resize_image_with_crop_or_pad(images, 108, 108)
    images = tf.image.resize_images(images, [64, 64])
    images = tf.cast(images, tf.float32) / 127.5 - 1.

    input_queue = tf.train.shuffle_batch([images], batch_size=batch_size, capacity=capacity,
                                         min_after_dequeue=min_after_dequeue)

    print "Train data loaded, num batch: %d" % num_batch

    return input_queue, num_batch

