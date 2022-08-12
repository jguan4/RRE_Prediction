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
# matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from RRETestProblem import RRETestProblem
from matplotlib.backends.backend_pdf import PdfPages
from networks import *
from utils import *
# order to networks: Psi, K, Theta
class RRENetwork(object):
	# struct: dictionary with entry layers (list) and toggle (string)
	def __init__(self, psistruct, Kstruct, thetastruct, training_hp, pathname):
		self.pathname = pathname
		self.norm = training_hp['norm']
		self.tfdtype = "float64"
		tf.keras.backend.set_floatx(self.tfdtype)
		self.training_hp = training_hp
		self.adam_options = self.training_hp['adam_options']
		self.tf_epochs = self.adam_options['epoch']
		self.tf_optimizer = tf.keras.optimizers.Adam()
		self.lb = tf.constant(self.training_hp['lb'],dtype = self.tfdtype)
		self.ub = tf.constant(self.training_hp['ub'],dtype = self.tfdtype)
		self.psi_lb = tf.constant(self.training_hp['psi_lb'],dtype = self.tfdtype)
		self.psi_ub = tf.constant(self.training_hp['psi_ub'],dtype = self.tfdtype)
		self.Psi = PsiNetwork(psistruct, self.pathname)
		self.K = KNetwork(Kstruct, self.pathname)
		self.Theta = ThetaNetwork(thetastruct, self.pathname)
		self.weights = self.training_hp['weights']
		self.psiweight_original = self.convert_tensor(self.weights[0])
		self.fweight_original = self.convert_tensor(self.weights[1])
		self.thetaweight = self.convert_tensor(self.weights[2])
		self.fluxweight_original = self.convert_tensor(self.weights[3])
		self.loss_log = [['Epoch'],['l_theta'],['l_f'],['l_top'],['l_bottom'],['weight_theta'],['weight_f'],['weight_tb'],['weight_bb']] # theta, residual, top bound, lower bound
		self.log_time = time.time()
		self.wrap_variable_sizes()
		self.starting_epoch = self.training_hp['starting_epoch']
		self.epoch = self.starting_epoch
		self.scheduleing_toggle = self.training_hp['scheduling_toggle']
		self.total_epoch = self.training_hp['total_epoch']

	@tf.function
	def rre_model(self, z_tf, t_tf):

		with tf.GradientTape(persistent=True) as tape:
			# Watching the two inputs we’ll need later, x and t
			tape.watch(z_tf)
			tape.watch(t_tf)

			# Packing together the inputs
			X_f = tf.squeeze(tf.stack([z_tf, t_tf],axis = 1))
			if self.norm == '_norm':
				X_f = 2*(X_f - self.lb)/(self.ub - self.lb) -1
			elif self.norm == '_norm1':
				X_f = (X_f - self.lb)/(self.ub - self.lb)

			# Getting the prediction
			psi = self.Psi.net(X_f)
			log_h = tf.math.log(-psi)
			theta = self.Theta.net(-log_h)
			K = self.K.net(-log_h)
			# Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
			psi_z = tape.gradient(psi, z_tf)
			psi_t = tape.gradient(psi, t_tf)
			theta_t = tape.gradient(theta, t_tf)
			K_z = tape.gradient(K,z_tf)

		# Getting the other derivatives
		psi_zz = tape.gradient(psi_z, z_tf)
		flux = -K*(psi_z+1)
		f_residual =  theta_t - K_z*psi_z- K*psi_zz - K_z

		return psi, K, theta, f_residual, flux, [psi_z, psi_t, theta_t, K_z, psi_zz]

	@tf.function
	def rre_model_pred(self, z_tf, t_tf):
		# Packing together the inputs
		X_f = tf.squeeze(tf.stack([z_tf, t_tf],axis = 1))
		if self.norm == '_norm':
			X_f = 2*(X_f - self.lb)/(self.ub - self.lb) -1
		elif self.norm == '_norm1':
			X_f = (X_f - self.lb)/(self.ub - self.lb)

		# Getting the prediction
		psi = self.Psi.net(X_f)
		log_h = tf.math.log(-psi)
		theta = self.Theta.net(-log_h)
		K = self.K.net(-log_h)
			
		return psi, K, theta

	def weight_scheduling(self, loss_theta = None):
		if self.scheduleing_toggle == 'constant':
			return self.thetaweight, self.fweight_original, self.psiweight_original, self.fluxweight_original
		elif self.scheduleing_toggle == 'linear':
			if self.epoch % 100 == 0 or self.epoch==self.starting_epoch:
				self.fweight = self.linear_shedule(start_weight=self.fweight_original, end_weight = self.convert_tensor(1.0), passed_epoch=self.epoch, duration = 5000) 
				self.psiweight = self.psiweight_original
				self.fluxweight = self.fluxweight_original
			return self.thetaweight,self.fweight,self.psiweight, self.fluxweight
		elif self.scheduleing_toggle == 'exp':
			if self.epoch %100 == 0 or self.epoch==self.starting_epoch:
				self.fweight = self.exp_schedule(start_weight=self.fweight_original, end_weight = self.convert_tensor(1.0), passed_epoch=self.epoch, duration = 5000)
			return self.thetaweight,self.fweight*10,self.psiweight, self.fluxweight

	def exp_schedule(self,start_weight,end_weight,passed_epoch,duration):
		y = start_weight*np.exp(np.log(end_weight/start_weight)/duration*passed_epoch) if passed_epoch<= duration else end_weight
		return y

	def linear_shedule(self, start_weight, end_weight, passed_epoch, duration):
		y = (end_weight-start_weight)/duration*passed_epoch+start_weight if passed_epoch<= duration else end_weight
		return y

	def loss(self, theta_data, residual_data, boundary_data, log = False, func_residual_toggle = False):
		ltheta =self.loss_theta(theta_data, log = log)
		lf =  self.loss_residual(residual_data, log = log)
		lb = self.loss_boundary(boundary_data, log = log)
		thetaweight, fweight, psiweight, fluxweight = self.weight_scheduling(loss_theta = ltheta)
		loss = ltheta*thetaweight+fweight*lf+lb
			
		if log:
			self.loss_log[5].append(thetaweight.numpy())
			self.loss_log[6].append(fweight.numpy())
		return loss

	def loss_theta(self, theta_data, log = False):
		_, _, theta_pred, _, _, _ = self.rre_model(theta_data['z'], theta_data['t'])
		loss = self.loss_reduce_mean(theta_pred, theta_data['data'])
		if log:
			self.loss_log[1].append(loss.numpy())
		return loss 

	def loss_residual(self, residual_data, log = False):
		_, _, _, f_pred, _, _ = self.rre_model(residual_data['z'], residual_data['t'])
		if log:
			self.loss_log[2].append(loss.numpy())
		return loss

	def loss_boundary(self, boundary_data, log = False):
		top_bound = boundary_data['top']
		bottom_bound = boundary_data['bottom']
		top_loss = self.loss_boundary_data(top_bound, log = log)
		bottom_loss = self.loss_boundary_data(bottom_bound, log = log)
		if log:
			self.loss_log[3].append(top_loss.numpy())
			self.loss_log[4].append(bottom_loss.numpy())
		return top_loss+bottom_loss

	def loss_boundary_data(self, bound, log = False):
		psi_pred, K_pred, theta_pred, f_pred, flux_pred, [psiz_pred, psit_pred, thetat_pred, Kz_pred, psizz_pred] = self.rre_model(bound['z'], bound['t'])
		_,_,psiweight,fluxweight = self.weight_scheduling()
		if bound['type'] == 'flux':
			loss = self.loss_reduce_mean(flux_pred, bound['data'])*fluxweight
			if log:
				self.loss_log[7].append(fluxweight.numpy())
		elif bound['type'] == 'psiz':
			loss = self.loss_reduce_mean(psiz_pred, bound['data'])
		elif bound['type'] == 'psi':
			loss = self.loss_reduce_mean(psi_pred, bound['data'])/(self.psi_ub-self.psi_lb)**2*psiweight
			if log:
				self.loss_log[8].append(psiweight.numpy())
		return loss

	@tf.function
	def loss_reduce_mean(self, pred, data):
		l = tf.reduce_mean(tf.square(data - pred))
		return l 

	@tf.function
	def loss_f(self, f):
		l = tf.reduce_mean(tf.square(f))
		return l

	# @tf.function
	def grad(self, theta_data, residual_data, boundary_data, flatten = False):
		with tf.GradientTape() as tape:
			loss_value = self.loss(theta_data, residual_data, boundary_data, log = True)
		grads = tape.gradient(loss_value, self.wrap_trainable_variables())
		return loss_value, grads

	def get_loss_and_flat_grad(self, theta_data, residual_data, boundary_data):
		def loss_and_flat_grad(w, log = False):
			with tf.GradientTape() as tape:
				self.set_weights(w)
				loss_value = self.loss(theta_data, residual_data, boundary_data, log = log)
			grad = tape.gradient(loss_value, self.wrap_trainable_variables())
			grad_flat = []
			for g in grad:
				grad_flat.append(tf.reshape(g, [-1]))
			grad_flat = tf.concat(grad_flat, 0)
			return loss_value, grad_flat.numpy()
		return loss_and_flat_grad	

	def fit(self, theta_data, residual_data, boundary_data):
		# convert samples to tensors for easy computation
		theta_data = self.convert_bound_data(theta_data)
		residual_data = self.convert_residual_data(residual_data)

		boundary_data['top'] = self.convert_bound_data(boundary_data['top'])
		boundary_data['bottom'] = self.convert_bound_data(boundary_data['bottom'])
			
		# Optimizing
		self.tf_optimization(theta_data, residual_data, boundary_data)
		self.sopt_optimization(theta_data, residual_data, boundary_data)


	def tf_optimization(self, theta_data, residual_data, boundary_data):
		for epoch in range(self.tf_epochs):
			if epoch %50 == 0 or epoch == self.tf_epochs:
				self.save_model()
				self.save_model(tag = self.epoch)
				self.save_loss()
			self.loss_log[0].append(self.epoch)
			loss_value = self.tf_optimization_step(theta_data, residual_data, boundary_data)
			print("Epoch {0}, loss value: {1}\n".format(epoch, loss_value))
			self.epoch += 1 

	def tf_optimization_step(self, theta_data, residual_data, boundary_data):
		loss_value, grads = self.grad(theta_data, residual_data, boundary_data)
		self.tf_optimizer.apply_gradients(
					zip(grads, self.wrap_trainable_variables()))
		return loss_value

	def sopt_optimization(self, theta_data, residual_data, boundary_data):
		x0 = self.get_weights()
		self.loss_and_flat_grad = self.get_loss_and_flat_grad(theta_data, residual_data, boundary_data)
		self.Nfeval = 0
		sopt.minimize(fun=self.loss_and_flat_grad, x0=x0, jac=True, method='L-BFGS-B', options = self.training_hp['lbfgs_options'], callback = self.sopt_callback)
		self.save_model()

	def sopt_callback(self,Xi):
		self.loss_log[0].append(self.epoch)
		self.loss_and_flat_grad(Xi, log = True)
		if self.Nfeval %50 == 0:
			self.save_model()
			self.save_model(tag = self.epoch)
			self.save_loss()
		self.Nfeval += 1
		self.epoch += 1

	def predict(self, z_tf, t_tf):
		z_tf = self.convert_tensor(z_tf)
		t_tf = self.convert_tensor(t_tf)
		psi, K, theta = self.rre_model_pred(z_tf, t_tf)
		return psi.numpy(), K.numpy(), theta.numpy()

	def normalize_psi(self, psi):
		psi = (psi-self.psi_lb)/(self.psi_ub-self.psi_lb)
		return psi

	def denormalize_psi(self,psi):
		psi = psi*(self.psi_ub-self.psi_lb)+self.psi_lb
		return psi

	def save_model(self, tag = ''):
		self.Psi.save_model(tag = tag)
		self.K.save_model(tag = tag)
		self.Theta.save_model(tag = tag)
	
	def save_loss(self):
		np.savetxt("{0}/loss_{1}.csv".format(self.pathname,self.log_time),  np.column_stack(self.loss_log), delimiter =", ", fmt ='% s')

	def wrap_trainable_variables(self):
		psi_vars = self.Psi.net.trainable_variables
		K_vars = self.K.net.trainable_variables
		theta_vars = self.Theta.net.trainable_variables
		variables = psi_vars+K_vars+theta_vars
		return variables

	def wrap_variable_sizes(self):
		psi_w = self.Psi.size_w
		psi_b = self.Psi.size_b
		K_w = self.K.size_w
		K_b = self.K.size_b
		theta_w = self.Theta.size_w
		theta_b = self.Theta.size_b
		self.sizes_w = psi_w+K_w+theta_w
		self.sizes_b = psi_b+K_b+theta_b

	def wrap_layers(self):
		psi_layers = self.Psi.net.layers
		K_layers = self.K.net.layers
		theta_layers = self.Theta.net.layers
		layers = psi_layers+K_layers+theta_layers
		return layers

	def get_weights(self, convert_to_tensor=True):
		w = []
		for layer in self.wrap_layers():
			weights_biases = layer.get_weights()
			weights = weights_biases[0].flatten()
			biases = weights_biases[1]
			w.extend(weights)
			w.extend(biases)
		if convert_to_tensor:
			w = self.convert_tensor(w)
		return w

	def set_weights(self, w):
		for i, layer in enumerate(self.wrap_layers()):
			start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
			end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
			weights = w[start_weights:end_weights]
			w_div = int(self.sizes_w[i] / self.sizes_b[i])
			weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
			biases = w[end_weights:end_weights + self.sizes_b[i]]
			weights_biases = [weights, biases]
			layer.set_weights(weights_biases)

	def load_model(self):
		self.Psi.load_model()
		self.K.load_model()
		self.Theta.load_model()

	def convert_tensor(self, data):
		data_tf = tf.convert_to_tensor(data, dtype=self.tfdtype)
		return data_tf

	def convert_bound_data(self, bound):
		bound['z'] = self.convert_tensor(bound['z'])
		bound['t'] = self.convert_tensor(bound['t'])
		bound['data'] = self.convert_tensor(bound['data'])
		return bound

	def convert_residual_data(self, residual):
		residual['z'] = self.convert_tensor(residual['z'])
		residual['t'] = self.convert_tensor(residual['t'])
		return residual
