import numpy as np
import os

_root = os.path.dirname(os.path.realpath(__file__))

train_labels = np.fromfile(os.path.join(_root, "MNIST", "raw", "train-labels-idx1-ubyte"), dtype=np.uint8)[8:]
train_images = np.fromfile(os.path.join(_root, "MNIST", "raw", "train-images-idx3-ubyte"), dtype=np.uint8)[16:].reshape([-1,28,28])
test_labels = np.fromfile(os.path.join(_root, "MNIST", "raw", "t10k-labels-idx1-ubyte"), dtype=np.uint8)[8:]
test_images = np.fromfile(os.path.join(_root, "MNIST", "raw", "t10k-images-idx3-ubyte"), dtype=np.uint8)[16:].reshape([-1,28,28])

def draw(label, image):
  for x in image:
    print(''.join([('%d' % label) if v > 0 else '*' for v in x]))

def draw_all():
  for label, image in zip(test_labels, test_images):
    draw(label, image)

if __name__ == "__main__":
  draw_all()
