import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

sess = tf.compat.v1.InteractiveSession()

from parse_mnist import *
import tqdm

image_ph = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None, None, 3))
encode_op = tf.io.encode_base64(tf.image.encode_png(image_ph))

with open('mnist_train.csv', 'w') as f:
  f.write('id,tag,width,height,channels,format,data\n')
  for i, [label, image] in tqdm.tqdm(list(enumerate(zip(train_labels, train_images))), desc="mnist_train.csv"):
    data = sess.run(encode_op, {image_ph: as_rgb(image)})
    f.write('%d,%d,%d,%d,3,png,%s\n' % (i, label, image.shape[-2], image.shape[-1], data.decode('utf8')))

with open('mnist_test.csv', 'w') as f:
  f.write('id,tag,width,height,channels,format,data\n')
  for i, [label, image] in tqdm.tqdm(list(enumerate(zip(test_labels, test_images))), desc="mnist_test.csv"):
    data = sess.run(encode_op, {image_ph: as_rgb(image)})
    f.write('%d,%d,%d,%d,3,png,%s\n' % (i, label, image.shape[-2], image.shape[-1], data.decode('utf8')))
