from PIL import Image
import os
import face_recognition
import argparse


def main(mode):
	if mode == 'train':
		filename = 'train'
	elif mode == 'test':
		filename = 'test'
	lists = os.listdir(filename)
	for item in lists:
		img_name = os.path.basename(item)
		if os.path.splitext(img_name)[1] != '.jpg':
			continue
		print(img_name)

		image = face_recognition.load_image_file(os.path.join(filename, img_name))
		face_location = face_recognition.face_locations(image)

		for location in face_location:
			top, right, bottom, left = location
			width = right - left
			height = bottom - top
			if width > height:
				right -= (width - height)
			elif height > width:
				bottom -= (height - width)
			face_image = image[top:bottom, left:right]
			pil_image = Image.fromarray(face_image)
			pil_image = pil_image.resize((64, 64), Image.ANTIALIAS)
			pil_image.save(filename + '_image/%s' % img_name)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
	config = parser.parse_args()
	main(config.mode)
