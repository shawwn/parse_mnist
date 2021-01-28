import numpy as np
import os

_root = os.path.dirname(os.path.realpath(__file__))

train_labels = np.fromfile(os.path.join(_root, "MNIST", "raw", "train-labels-idx1-ubyte"), dtype=np.uint8)[8:]
train_images = np.fromfile(os.path.join(_root, "MNIST", "raw", "train-images-idx3-ubyte"), dtype=np.uint8)[16:].reshape([-1,28,28])
test_labels = np.fromfile(os.path.join(_root, "MNIST", "raw", "t10k-labels-idx1-ubyte"), dtype=np.uint8)[8:]
test_images = np.fromfile(os.path.join(_root, "MNIST", "raw", "t10k-images-idx3-ubyte"), dtype=np.uint8)[16:].reshape([-1,28,28])

def as_rgb(image):
  if len(image.shape) == 2:
    return np.repeat(train_images[1], 3).reshape(list(image.shape) + [3])
  if len(image.shape) == 3:
    return image
  assert False

def draw(label, image):
  for x in image:
    print(''.join([('%d' % label) if any(v > 0) else '*' for v in x]))

def draw_all():
  for label, image in zip(test_labels, test_images):
    draw(label, image)

def maketree(path):
  try:
    os.makedirs(path)
  except:
    pass

def save_all():
  import tensorflow as tf
  import tensorflow.compat.v1 as tf1
  import tqdm
  sess = tf1.InteractiveSession()
  image_ph = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None, None, 3))
  encode_op = tf.image.encode_png(image_ph)
  maketree('mnistimg/mnist')
  with open('mnistimg/mnist-validation.txt', 'w') as filenames:
    for i in tqdm.trange(len(test_images)):
      label = test_labels[i]
      image = test_images[i]
      path = "mnist/validation/{}/{}.png".format(label, i)
      filenames.write(path + '\n')
      filenames.flush()
      path = "mnistimg/" + path
      maketree(os.path.dirname(path))
      data = sess.run(encode_op, {image_ph: as_rgb(image)})
      with open(path, "wb") as f:
        f.write(data)
  with open('mnistimg/mnist-train.txt', 'w') as filenames:
    for i in tqdm.trange(len(train_images)):
      label = train_labels[i]
      image = train_images[i]
      path = "mnist/train/{}/{}.png".format(label, i)
      filenames.write(path + '\n')
      filenames.flush()
      path = "mnistimg/" + path
      maketree(os.path.dirname(path))
      data = sess.run(encode_op, {image_ph: as_rgb(image)})
      with open(path, "wb") as f:
        f.write(data)

if __name__ == "__main__":
  save_all()
