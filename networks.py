import numpy as np
import math
import time
import sys
import csv
import pandas as pd
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import scipy.optimize as sopt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(5)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

class PsiNetwork(object):
	def __init__(self,psistruct, pathname):
		self.layers = psistruct['layers']
		self.NN_toggle = psistruct['toggle']
		N_hidden = len(self.layers)-2
		N_width = self.layers[1]
		self.size_w = []
		self.size_b = []
		self.path = "{2}/Psi{3}_{0}layer_{1}width_checkpoint".format(N_hidden, N_width, pathname, self.NN_toggle)
		self.initialize_PsiNN()
		self.get_weights_struct()

	def initialize_PsiNN(self):
		if self.NN_toggle == 'DNN':
			self.initialize_DNN()
		elif self.NN_toggle == 'ResNet':
			self.initialize_ResNet()

	def initialize_DNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer="glorot_normal"))
		self.net.add(PsiOutput(self.layers[-1]))

	def initialize_ResNet(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		self.net.add(tf.keras.layers.Dense(
			self.layers[1], activation=tf.nn.tanh,
			kernel_initializer="glorot_normal"))
		for width in self.layers[2:-1]:
			self.net.add(ResLayer(width))
		self.net.add(PsiOutput(self.layers[-1]))

	def get_weights_struct(self):
		for i in range(len(self.net.layers)):
			self.size_w.append(tf.size(self.net.layers[i].kernel))
			self.size_b.append(tf.size(self.net.layers[i].bias))

	def save_model(self, tag = ''):
		self.net.save(self.path+"{0}.h5".format(tag))

	def load_model(self):
		self.net.load_weights(self.path+'.h5')

class KNetwork(object):
	def __init__(self,Kstruct, pathname):
		self.layers = Kstruct['layers']
		self.size_w = []
		self.size_b = []
		N_hidden = len(self.layers)-2
		N_width = self.layers[1]
		self.NN_toggle = Kstruct['toggle']
		self.path = "{2}/K{3}_{0}layer_{1}width_checkpoint".format(N_hidden, N_width, pathname, self.NN_toggle)
		if self.NN_toggle == 'MNN':
			self.initialize_KMNN()
		elif self.NN_toggle == 'ResNet':
			self.initialize_KResNN()
		else:
			self.initialize_KNN()
		self.get_weights_struct()

	def get_weights_struct(self):
		for i in range(len(self.net.layers)):
			self.size_w.append(tf.size(self.net.layers[i].kernel))
			self.size_b.append(tf.size(self.net.layers[i].bias))

	def initialize_KMNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer=NMNN_Init))
		self.net.add(KOutput(self.layers[-1]))

	def initialize_KNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer="glorot_normal"))
		self.net.add(KOutput(self.layers[-1]))

	def initialize_KResNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		self.net.add(tf.keras.layers.Dense(
			self.layers[1], activation=tf.nn.tanh,
			kernel_initializer="glorot_normal"))
		for width in self.layers[2:-1]:
			self.net.add(ResLayer(
				width))
		self.net.add(KOutput(self.layers[-1]))

	def save_model(self, tag = ''):
		self.net.save_weights(self.path+"{0}.h5".format(tag))

	def load_model(self):
		self.net.load_weights(self.path+'.h5')

class ThetaKNetwork(object):
	def __init__(self,ThetaKstruct, pathname):
		self.layers = ThetaKstruct['layers']
		self.size_w = []
		self.size_b = []
		N_hidden = len(self.layers)-2
		N_width = self.layers[1]
		self.NN_toggle = ThetaKstruct['toggle']
		self.path = "{2}/ThetaK{3}_{0}layer_{1}width_checkpoint".format(N_hidden, N_width, pathname, self.NN_toggle)
		if self.NN_toggle == 'MNN':
			self.initialize_MNN()
		elif self.NN_toggle == 'ResNet':
			self.initialize_ResNN()
		else:
			self.initialize_NN()
		self.get_weights_struct()

	def get_weights_struct(self):
		for i in range(len(self.net.layers)):
			self.size_w.append(tf.size(self.net.layers[i].kernel))
			self.size_b.append(tf.size(self.net.layers[i].bias))

	def initialize_MNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer=NMNN_Init))
		self.net.add(ThetaKOutput(self.layers[-1]))

	def initialize_NN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer="glorot_normal"))
		self.net.add(ThetaKOutput(self.layers[-1]))

	def initialize_ResNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		self.net.add(tf.keras.layers.Dense(
			self.layers[1], activation=tf.nn.tanh,
			kernel_initializer="glorot_normal"))
		for width in self.layers[2:-1]:
			self.net.add(ResLayer(
				width))
		self.net.add(ThetaKOutput(self.layers[-1]))

	def save_model(self, tag = ''):
		self.net.save_weights(self.path+"{0}.h5".format(tag))

	def load_model(self):
		self.net.load_weights(self.path+'.h5')

class ThetaNetwork(object):
	def __init__(self,thetastruct, pathname):
		self.layers = thetastruct['layers']
		self.size_w = []
		self.size_b = []
		N_hidden = len(self.layers)-2
		N_width = self.layers[1]
		self.NN_toggle = thetastruct['toggle']
		self.path = "{2}/Theta{3}_{0}layer_{1}width_checkpoint".format(N_hidden, N_width, pathname, self.NN_toggle)
		if self.NN_toggle == 'MNN':
			self.initialize_ThetaMNN()
		elif self.NN_toggle == 'ResNet':
			self.initialize_ThetaResNN()
		else:
			self.initialize_ThetaNN()
		self.get_weights_struct()

	def get_weights_struct(self):
		for i in range(len(self.net.layers)):
			self.size_w.append(tf.size(self.net.layers[i].kernel))
			self.size_b.append(tf.size(self.net.layers[i].bias))

	def initialize_ThetaNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer="glorot_normal"))
		self.net.add(tf.keras.layers.Dense(
				self.layers[-1], activation=tf.nn.sigmoid,
				kernel_initializer="glorot_normal"),bias_initializer = tf.keras.initializers.Constant(0))

	def initialize_ThetaResNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		self.net.add(tf.keras.layers.Dense(
			self.layers[1], activation=tf.nn.tanh,
			kernel_initializer="glorot_normal"))
		for width in self.layers[2:-1]:
			self.net.add(ResLayer(
				width))
		self.net.add(tf.keras.layers.Dense(
				self.layers[-1], activation=tf.nn.sigmoid,
				kernel_initializer="glorot_normal",bias_initializer = tf.keras.initializers.Constant(0)))

	def initialize_ThetaMNN(self):
		self.net = tf.keras.Sequential()
		self.net.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
		for width in self.layers[1:-1]:
			self.net.add(tf.keras.layers.Dense(
				width, activation=tf.nn.tanh,
				kernel_initializer=MNN_Init))
		self.net.add(tf.keras.layers.Dense(
				self.layers[-1], activation=tf.nn.sigmoid,
				kernel_initializer=MNN_Init,bias_initializer = tf.keras.initializers.Constant(0)))

	def save_model(self, tag = ''):
		self.net.save_weights(self.path+"{0}.h5".format(tag))

	def load_model(self):
		self.net.load_weights(self.path+'.h5')


class PsiOutput(tf.keras.layers.Layer):
	def __init__(self, units=32):
		super(PsiOutput, self).__init__()
		self.units = units

	def build(self, input_shape):
		self.kernel = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer="glorot_normal",
			trainable=True, name = "kernel")
		self.bias = self.add_weight(
			shape=(self.units,), initializer="zeros", trainable=True, name = "bias")

	def call(self, inputs):
		return -tf.math.exp(tf.matmul(inputs, self.kernel) + self.bias)

	def get_config(self):
		return {"units": self.units}


class KOutput(tf.keras.layers.Layer):
	def __init__(self, units=32):
		super(KOutput, self).__init__()
		self.units = units

	def build(self, input_shape):
		self.kernel = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=MNN_Init,
			trainable=True,name = "kernel")
		self.bias = self.add_weight(
			shape=(self.units,), initializer=tf.keras.initializers.Constant(0), trainable=True, name = "bias")

	def call(self, inputs):
		return tf.math.exp(-(tf.matmul(inputs, self.kernel) + self.bias))

	def get_config(self):
		return {"units": self.units}

class ThetaKOutput(tf.keras.layers.Layer):
	def __init__(self, units=32):
		super(ThetaKOutput, self).__init__()
		self.units = units

	def build(self, input_shape):
		self.kernel = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer="glorot_normal",
			trainable=True,name = "kernel")
		self.bias = self.add_weight(
			shape=(self.units,), initializer="zeros", trainable=True, name = "bias")

	def call(self, inputs):
		return tf.matmul(inputs, self.kernel) + self.bias

	def get_config(self):
		return {"units": self.units}


class MNN_Init(tf.keras.initializers.Initializer):

	def __init__(self):
		self.mean = 0 
		self.stddev = 0

	def __call__(self, shape, dtype=None, **kwargs):
		in_dim = shape[0]
		out_dim = shape[1]
		self.stddev = np.sqrt(2/(in_dim + out_dim))
		W = tf.random.normal(shape, stddev=self.stddev, dtype=dtype)
		return W**2

	def get_config(self):  # To support serialization
		return {"mean": self.mean, "stddev": self.stddev}

class NMNN_Init(tf.keras.initializers.Initializer):

	def __init__(self):
		self.mean = 0 
		self.stddev = 0

	def __call__(self, shape, dtype=None, **kwargs):
		in_dim = shape[0]
		out_dim = shape[1]
		self.stddev = np.sqrt(2/(in_dim + out_dim))
		W = tf.random.normal(shape, stddev=self.stddev, dtype=dtype)
		return -W**2

	def get_config(self):  # To support serialization
		return {"mean": self.mean, "stddev": self.stddev}


class ResLayer(tf.keras.layers.Layer):
	def __init__(self, units=32):
		super(ResLayer, self).__init__()
		self.units = units

	def build(self, input_shape):
		self.kernel = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer="glorot_normal",
			trainable=True,name = "kernel")
		self.bias = self.add_weight(
			shape=(self.units,), initializer="zeros", trainable=True, name = "bias")

	def call(self, inputs):
		return tf.math.tanh(tf.matmul(inputs, self.kernel) + self.bias)+0.5*inputs


	def get_config(self):
		return {"units": self.units}