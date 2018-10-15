import tensorflow as tf
from util import *

# wgan
LEARNING_RATE = 0.0002
BETA1 = 0.5
EPOCH = 20
BATCH_SIZE = 64
DATASET = 'train_image'
CHECKPOINT_DIR_v2 = 'checkpoint_v2'
SAMPLE_DIR = 'samples'

# completion
LAMBDA = 0.1
CENTER_SCALE = 0.25  # should <= 0.5
COMPLETION_DIR = 'complete'
NUM_ITER = 1000
C_BETA1 = 0.9
C_BETA2 = 0.999
LR = 0.01
EPS = 1e-8
UNCOMPLETION_IMAGE_DIR = 'test_image'


class Generator(object):
	def __init__(self, s_size=4):
		self.s_size = s_size
		self.reuse = False
		self.name = 'generator'

	def __call__(self, inputs, is_training=False):
		inputs = tf.convert_to_tensor(inputs)
		inputs = tf.cast(inputs, tf.float32)

		with tf.variable_scope(self.name, reuse=self.reuse) as vs:
			outputs = tf.layers.dense(
				inputs,
				512 * self.s_size * self.s_size,
				kernel_initializer=tf.random_normal_initializer(stddev=0.02)
			)
			outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, 512])
			outputs = tf.nn.relu(
				tf.contrib.layers.batch_norm(
					outputs,
					decay=0.9,
					updates_collections=None,
					epsilon=1e-5,
					center=True,
					scale=True,
					is_training=is_training
				), name='fc'
			)

			outputs = tf.layers.conv2d_transpose(
				outputs,
				256,
				[5, 5],
				strides=(2, 2),
				padding='SAME',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02)
			)
			outputs = tf.nn.relu(
				tf.contrib.layers.batch_norm(
					outputs,
					decay=0.9,
					updates_collections=None,
					epsilon=1e-5,
					center=True,
					scale=True,
					is_training=is_training
				), name='deconv1'
			)

			outputs = tf.layers.conv2d_transpose(
				outputs,
				128,
				[5, 5],
				strides=(2, 2),
				padding='SAME',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02)
			)
			outputs = tf.nn.relu(
				tf.contrib.layers.batch_norm(
					outputs,
					decay=0.9,
					updates_collections=None,
					epsilon=1e-5,
					center=True,
					scale=True,
					is_training=is_training
				), name='deconv2'
			)

			outputs = tf.layers.conv2d_transpose(
				outputs,
				64,
				[5, 5],
				strides=(2, 2),
				padding='SAME',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02)
			)
			outputs = tf.nn.relu(
				tf.contrib.layers.batch_norm(
					outputs,
					decay=0.9,
					updates_collections=None,
					epsilon=1e-5,
					center=True,
					scale=True,
					is_training=is_training
				), name='deconv3'
			)

			outputs = tf.layers.conv2d_transpose(
				outputs,
				3,
				[5, 5],
				strides=(2, 2),
				padding='SAME',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02)
			)
			outputs = tf.tanh(outputs, name='outputs')

		self.reuse = True
		self.var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
		return outputs


class Discriminator(object):
	def __init__(self):
		self.reuse = False
		self.name = 'discriminator'

	def __call__(self, inputs, is_training=False):
		inputs = tf.convert_to_tensor(inputs)
		inputs = tf.cast(inputs, tf.float32)

		with tf.variable_scope(self.name, reuse=self.reuse) as vs:
			outputs = tf.layers.conv2d(
				inputs,
				64,
				[5, 5],
				strides=(2, 2),
				padding='SAME',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02)
			)
			outputs = tf.nn.leaky_relu(
				tf.contrib.layers.batch_norm(
					outputs,
					decay=0.9,
					updates_collections=None,
					epsilon=1e-5,
					center=True,
					scale=True,
					is_training=is_training
				), name='conv1'
			)

			outputs = tf.layers.conv2d(
				outputs,
				128,
				[5, 5],
				strides=(2, 2),
				padding='SAME',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02)
			)
			outputs = tf.nn.leaky_relu(
				tf.contrib.layers.batch_norm(
					outputs,
					decay=0.9,
					updates_collections=None,
					epsilon=1e-5,
					center=True,
					scale=True,
					is_training=is_training
				), name='conv2'
			)

			outputs = tf.layers.conv2d(
				outputs,
				256,
				[5, 5],
				strides=(2, 2),
				padding='SAME',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02)
			)
			outputs = tf.nn.leaky_relu(
				tf.contrib.layers.batch_norm(
					outputs,
					decay=0.9,
					updates_collections=None,
					epsilon=1e-5,
					center=True,
					scale=True,
					is_training=is_training
				), name='conv3'
			)

			outputs = tf.layers.conv2d(
				outputs,
				512,
				[5, 5],
				strides=(2, 2),
				padding='SAME',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02)
			)
			outputs = tf.nn.leaky_relu(
				tf.contrib.layers.batch_norm(
					outputs,
					decay=0.9,
					updates_collections=None,
					epsilon=1e-5,
					center=True,
					scale=True,
					is_training=is_training
				), name='conv4'
			)

			outputs = tf.reshape(outputs, [-1, 8192])
			outputs = tf.layers.dense(outputs, 1, name='outputs')

		self.reuse = True
		self.var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
		return outputs


class WGAN(object):
	def __init__(self):
		config = tf.ConfigProto(
			allow_soft_placement=True,
			log_device_placement=False
		)
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		self.batch_size = BATCH_SIZE
		self.z_dim = 100
		self.G = Generator()
		self.D = Discriminator()

		self.sample_size = 64
		self.model_name = 'WGAN.model'

		self.lambda_ = LAMBDA
		self.learning_rate = LEARNING_RATE

		self.checkpoint_dir = CHECKPOINT_DIR_v2
		self.build()

	def build(self):
		self.is_training = tf.placeholder(tf.bool, name='is_training')
		self.images = tf.placeholder(tf.float32, [None, 64, 64, 3], name='real_images')
		self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
		self.z_sum = tf.summary.histogram('z', self.z)

		# GAN
		self.g = self.G(self.z, is_training=self.is_training)
		self.d_logits = self.D(self.images, is_training=self.is_training)
		self.d_logits_ = self.D(self.g, is_training=self.is_training)

		self.g_sum = tf.summary.histogram('g', self.g)
		self.d_sum = tf.summary.histogram('d', self.d_logits)
		self.d_sum_ = tf.summary.histogram('d_', self.d_logits_)

		self.g_loss = -tf.reduce_mean(self.d_logits_)
		self.d_loss = -tf.reduce_mean(self.d_logits) + tf.reduce_mean(self.d_logits_)

		self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
		self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)

		trainable_vars = tf.trainable_variables()
		self.d_vars = [var for var in trainable_vars if 'discriminator' in var.name]
		self.g_vars = [var for var in trainable_vars if 'generator' in var.name]

		self.saver = tf.train.Saver(max_to_keep=5)

		self.d_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
		self.g_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.g_loss, var_list=self.g_vars)

		self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_vars]

	def train_gan(self):
		data = data_index(DATASET)
		self.sess.run(tf.global_variables_initializer())

		self.summary_op = tf.summary.merge(
			[self.z_sum, self.g_sum, self.d_sum_, self.g_loss_sum,
			 self.d_sum, self.d_loss_sum]
		)
		self.writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)

		eval_z = np.random.uniform(-1.0, 1.0, [self.sample_size, self.z_dim])
		eval_image = read_batch(data[0:self.sample_size], 'train_image')

		if load(self.checkpoint_dir, self.saver, self.sess):
			print('Loaded!')
		else:
			print('Failed to load checkpoint and init all variable!')

		count = 1
		for epoch in range(EPOCH):
			batch_index = len(data) // self.batch_size

			for index in range(batch_index):
				train_images = read_batch(data[index * self.batch_size:(index + 1) * self.batch_size], 'train_image')
				train_z = np.random.uniform(-1.0, 1.0, [self.batch_size, self.z_dim]).astype(np.float32)

				# update D and twice G to make sure d_loss > 0
				self.sess.run(
					[self.d_op],
					feed_dict={self.images: train_images, self.z: train_z, self.is_training: True}
				)
				self.sess.run(
					[self.d_clip]
				)
				self.sess.run(
					[self.g_op],
					feed_dict={self.z: train_z, self.is_training: True}
				)
				self.sess.run(
					[self.g_op],
					feed_dict={self.z: train_z, self.is_training: True}
				)

				err_d = self.sess.run(
					self.d_loss,
					feed_dict={self.z: train_z, self.images: train_images, self.is_training: False}
				)
				err_g = self.sess.run(
					self.g_loss,
					feed_dict={self.z: train_z, self.is_training: False}
				)

				count += 1
				print('Epoch: [{:2d}] [{:4d}/{:4d}], d_loss: {:.8f}, g_loss: {:.8f}'.format(
					epoch, index, batch_index, err_d, err_g))

				if count % 100 == 0:
					samples, d_loss, g_loss, summary_str = self.sess.run(
						[self.g, self.d_loss, self.g_loss, self.summary_op],
						feed_dict={self.z: eval_z, self.images: eval_image, self.is_training: False}
					)
					self.writer.add_summary(summary_str, count)
					save_images(samples, [8, 8], './samples/train_{:02d}_{:04d}.png'.format(epoch, count))
					print('[Eval] d_loss: {:.8f}, g_loss: {:.8f}'.format(d_loss, g_loss))

				if count % 500 == 0:
					self.save(self.checkpoint_dir, count)
					print('Save the checkpoint for step: {:4d}'.format(count))

	def save(self, checkpoint_dir, step):
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name), global_step=step)


class Completion(WGAN):
	def __init__(self):
		self.lambda_ = LAMBDA
		super(Completion, self).__init__()
		self.settings()

	def settings(self):
		self.lowers_g = tf.reduce_mean(
			tf.reshape(self.g, [self.batch_size, 8, 8, 8, 8, 3]),
			[2, 4]
		)
		self.lowers_images = tf.reduce_mean(
			tf.reshape(self.images, [self.batch_size, 8, 8, 8, 8, 3]),
			[2, 4]
		)

		self.mask = tf.placeholder(tf.float32, [64, 64, 3], name='mask')
		self.lowers_mask = tf.placeholder(tf.float32, [8, 8, 3], name='lowers_mask')

		self.contextual_loss = tf.reduce_sum(
			tf.layers.flatten(
				tf.abs(
					tf.multiply(self.mask, self.g) - tf.multiply(self.mask, self.images)
				)
			), 1
		) + tf.reduce_sum(
			tf.layers.flatten(
				tf.abs(
					tf.multiply(self.lowers_mask, self.lowers_g) - tf.multiply(self.lowers_mask, self.lowers_images)
				)
			), 1
		)
		self.perceptual_loss = self.g_loss
		self.complete_loss = self.contextual_loss + self.lambda_ * self.perceptual_loss
		self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

	def train_completion(self):
		if not os.path.exists(COMPLETION_DIR):
			os.makedirs(COMPLETION_DIR)
		p = os.path.join(COMPLETION_DIR, 'hats_imgs')
		if not os.path.exists(p):
			os.makedirs(p)
		p = os.path.join(COMPLETION_DIR, 'completed')
		if not os.path.exists(p):
			os.makedirs(p)
		self.sess.run(tf.global_variables_initializer())

		loaded = load(self.checkpoint_dir, self.saver, self.sess)
		assert loaded

		data = data_index(UNCOMPLETION_IMAGE_DIR, shuffle=True)
		data_num = len(data)

		batch_index = int(np.ceil(data_num / self.batch_size))
		lowers_mask = np.zeros([8, 8, 3])

		# center mask
		mask = np.ones([64, 64, 3])
		l = int(64 * CENTER_SCALE)
		u = int(64 * (1. - CENTER_SCALE))
		mask[l:u, l:u, :] = 0.0

		# randomly choose 64 images
		batch_size_z = min(self.batch_size, data_num)
		batch = read_batch(data[0:batch_size_z], 'test_image')

		if batch_size_z < self.batch_size:
			pad_size = ((0, int(self.batch_size - batch_size_z)), (0, 0), (0, 0), (0, 0))
			batch = np.pad(batch, pad_size, 'constant').astype(np.float32)

		z = np.random.uniform(-1.0, 1.0, [self.batch_size, self.z_dim])

		rows = np.ceil(batch_size_z / 8)
		cols = min(8, batch_size_z)
		save_images(batch[:batch_size_z, :, :, :], [rows, cols], os.path.join(COMPLETION_DIR, 'before.png'))
		masked_images = np.multiply(batch, mask)
		save_images(masked_images[:batch_size_z, :, :, :], [rows, cols], os.path.join(COMPLETION_DIR, 'masked.png'))

		# Adam Optimizer Params
		m = 0
		v = 0
		for i in range(NUM_ITER):
			loss, g, g_imgs, lowers_g_imgs = self.sess.run(
				[self.complete_loss, self.grad_complete_loss, self.g, self.lowers_g],
				feed_dict={
					self.z: z,
					self.mask: mask,
					self.lowers_mask: lowers_mask,
					self.images: batch,
					self.is_training: False
				}
			)

			if i % 50 == 0:
				print('Iter: [{:4d}/{:4d}], mean loss: {:.8f}, mean g: {:.8f}'.format(
					i, NUM_ITER, np.mean(loss[0:batch_size_z]), np.mean(g[0:batch_size_z])
				))
				img_name = os.path.join(COMPLETION_DIR, 'hats_imgs/{:04d}.png'.format(i))
				save_images(g_imgs[:batch_size_z, :, :, :], [rows, cols], img_name)

				inv_masked_hat_images = np.multiply(g_imgs, 1.0 - mask)
				completed = masked_images + inv_masked_hat_images

				img_name = os.path.join(COMPLETION_DIR, 'completed/{:04d}.png'.format(i))
				save_images(completed[:batch_size_z, :, :, :], [rows, cols], img_name)

			# AdamOptimizer
			m_prev = np.copy(m)
			v_prev = np.copy(v)
			m = C_BETA1 * m_prev + (1 - C_BETA1) * g[0]
			v = C_BETA2 * v_prev + (1 - C_BETA2) * np.multiply(g[0], g[0])
			m_ = m / (1 - C_BETA1 ** (i + 1))
			v_ = v / (1 - C_BETA2 ** (i + 1))
			z += np.true_divide(LR * m_, np.sqrt(v_) + EPS)
			z = np.clip(z, -1, 1)
