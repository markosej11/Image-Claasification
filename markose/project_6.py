import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob

train_dir = 'dogs-vs-cats/train'
test_dir = 'dogs-vs-cats/test1'
img_size = 50
lr = 0.001
model_name = 'dogsvscats-{}-{}.model'.format(lr, '4 layers - first_try - 1')

def lable_img(img):
	word_lable = img.split('.')[-3]
	if word_lable == 'cat' : return [1,0]
	elif word_lable == 'dog' : return [0,1]

def create_train_data():
	training_data = []
	for img in tqdm(os.listdir(train_dir)):
		lable = lable_img(img)
		path = os.path.join(train_dir,img)
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (img_size,img_size))
		training_data.append([np.array(img), np.array(lable)])
	shuffle(training_data)
	np.save('training_data.npy', training_data)
	return training_data

def process_test_data():
	testing_data = []
	for img in tqdm(os.listdir(test_dir)):
		path = os.path.join(test_dir,img)
		img_num = img.split('.')[0]
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (img_size,img_size))
		testing_data.append([np.array(img), img_num])
	np.save('test_data.npy', testing_data)
	return testing_data

train_data = create_train_data()
# train_data = np.load('training_data.npy',allow_pickle=True)

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')


model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=3)

if os.path.exists('{}.meta'.format(model_name)):
	model.load(model_name)
	print( "model loaded!")


train = train_data[:18000]
test = train_data

X = np.array([i[0] for i in train ]).reshape(-1, img_size, img_size, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test ]).reshape(-1, img_size, img_size, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
	snapshot_step=500, show_metric=True, run_id=model_name)


model.save(model_name)

test_data = process_test_data()
# test_data = np.load('test_data.npy',allow_pickle=True)

fig = plt.figure()

for num, data in enumerate(test_data[:12]):
	img_num = data[1]
	img_data = data[0]

	y = fig.add_subplot(3,4,num+1)
	orig = img_data
	data = img_data.reshape(img_size,img_size,1)

	model_out = model.predict([data])[0]

	if np.argmax(model_out) == 1: 
		str = "dog"
	else:
		str="cat"

	y.imshow(orig, cmap = 'gray')
	plt.title(str)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show()

with open('group-27_cat.cvs', 'w') as f:
	f.write('id,lable\n')
with open('group-27_dog.cvs', 'w') as f:
	f.write('id,lable\n')
with open('group-27_cat.cvs', 'a') as f:
	with open('group-27_dog.cvs', 'a') as g:
		loss = 0.0
		for data in tqdm(test_data):
			img_num = data[1]
			img_data = data[0]
			orig = img_data
			data = img_data.reshape(img_size,img_size,1)
			model_out = model.predict([data])[0]
			category = round(model_out[1])
			loss -= np.log(model_out[1]) * category

			(g if category else f).write('{},{}\n'.format(img_num, 0.5 + abs(model_out[1] - 0.5)))
		print("\nTESTING LOSS: %f" % loss)
