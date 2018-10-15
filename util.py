import os
import random
import scipy.misc
import numpy as np
import tensorflow as tf


def imread(path):
	return scipy.misc.imread(path, mode='RGB').astype(np.float32)


def data_index(data_dir, shuffle=False):
	dirs = os.listdir(data_dir)
	lists = []

	for item in dirs:
		image_name = os.path.basename(item)
		if os.path.splitext(image_name)[1] != '.jpg':
			continue
		lists.append(item)

	if shuffle:
		random.shuffle(lists)

	return lists


def read_batch(lists, dir):
	batch = [imread(dir + '/' + a) for a in lists]
	batch = np.array(batch).astype(np.float32)

	batch = batch / 127.5 - 1
	return batch


def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	img = np.zeros((int(h * size[0]), int(w * size[1]), 3))

	for index, image in enumerate(images):
		i = int(index // size[0])
		j = int(index % size[1])
		img[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = image

	return img


def save_images(images, size, image_path):
	images = (images + 1.) / 2
	img = merge(images, size)
	return scipy.misc.imsave(image_path, (255 * img).astype(np.uint8))


def load(ckpt_dir, saver, sess):
	ckpt = tf.train.get_checkpoint_state(ckpt_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		return True
	else:
		return False
