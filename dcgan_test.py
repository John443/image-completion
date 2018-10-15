from network import *
import argparse


def train_gan():
	if not os.path.exists(CHECKPOINT_DIR_v1):
		os.makedirs(CHECKPOINT_DIR_v1)
	if not os.path.exists(SAMPLE_DIR):
		os.makedirs(SAMPLE_DIR)

	gan = DCGAN()
	gan.train_gan()
	print('Finish to train dcgan')


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
