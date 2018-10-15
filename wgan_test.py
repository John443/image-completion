from network_wgan import *
import argparse


def train_gan():
	if not os.path.exists(CHECKPOINT_DIR_v2):
		os.makedirs(CHECKPOINT_DIR_v2)
	if not os.path.exists(SAMPLE_DIR):
		os.makedirs(SAMPLE_DIR)

	gan = WGAN()
	gan.train_gan()
	print('Finish to train wgan')


def train_completion():
	completion = Completion()
	completion.train_completion()
	print('Finish to complete images')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='gan', choices=['gan', 'complete'])
	config = parser.parse_args()

	if config.mode == 'gan':
		train_gan()
	elif config.mode == 'complete':
		train_completion()
