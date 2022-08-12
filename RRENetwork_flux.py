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
from RRETestProblem import RRETestProblem
from RRENetwork import *
from matplotlib.backends.backend_pdf import PdfPages
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
from utils import *
from networks import *

# input (z,t,q) 
class RRENetwork_flux(RRENetwork):
	def __init__(self, psistruct, Kstruct, thetastruct, training_hp, pathname):
		super().__init__(psistruct, Kstruct, thetastruct, training_hp, pathname)
		self.loss_log = [['Epoch'],['l_theta'],['l_f'],['l_top'],['l_bottom'],['weight_theta'],['weight_f'],['weight_tb'],['weight_lb'],['l_diff']] # theta, residual, top bound, lower bound, discrete residual

	def rre_model(self, z_tf, t_tf, flux_tf):

		with tf.GradientTape(persistent=True) as tape:
			# Watching the two inputs we’ll need later, z, t and flux
			tape.watch(z_tf)
			tape.watch(t_tf)
			tape.watch(flux_tf)

			# Packing together the inputs
			X_f = tf.squeeze(tf.stack([z_tf, t_tf, flux_tf],axis = 1))
			if self.norm == '_norm':
				X_f = 2*(X_f - self.lb)/(self.ub - self.lb) -1
			elif self.norm == '_norm1':
				X_f = (X_f - self.lb)/(self.ub - self.lb)
			psi = self.Psi.net(X_f)
			log_h = tf.math.log(-psi)
			theta = self.Theta.net(-log_h)
			K = self.K.net(-log_h)
			# Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
			psi_z = tape.gradient(psi, z_tf)
			flux = -K*(psi_z+1)
			psi_t = tape.gradient(psi, t_tf)
			theta_t = tape.gradient(theta, t_tf)
			K_z = tape.gradient(K,z_tf)

		# Getting the other derivatives
		psi_zz = tape.gradient(psi_z, z_tf)
		flux_residual = flux-flux_tf
		f_residual =  theta_t - K_z*psi_z- K*psi_zz - K_z
		del tape

		return psi, K, theta, f_residual, flux, [psi_z, psi_t, theta_t, K_z, psi_zz, flux_residual]

	def rre_model_pred(self, z_tf, t_tf, flux_tf):

		# Packing together the inputs
		X_f = tf.squeeze(tf.stack([z_tf, t_tf, flux_tf],axis = 1))
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

	def loss_theta(self, theta_data, log = False):
		_, _, theta_pred, _, _, [_, _, _, _, _, _] = self.rre_model(theta_data['z'], theta_data['t'], theta_data['flux'])
		loss = self.loss_reduce_mean(theta_pred, theta_data['data'])
		if log:
			self.loss_log[1].append(loss.numpy())
		return loss 

	def loss_residual(self, residual_data, log = False):
		_, _, theta_pred, f_pred, _, [_, _, theta_t, _, _, _] = self.rre_model(residual_data['z'], residual_data['t'], residual_data['flux'])

		# compute discrete residual, does require the mesh in depth to be the same across times
		nz = residual_data['nz']
		t_r = tf.reshape(residual_data['t'], [int(len(residual_data['t'])/nz),nz])
		dt = t_r[1::,:]-t_r[0:-1,:]
		thetat_r = tf.reshape(theta_t, [int(len(theta_t)/nz),nz])
		theta_r = tf.reshape(theta_pred, [int(len(theta_pred)/nz),nz])
		backward_theta_pred = theta_r[1::,:]-dt*thetat_r[1::,:]
		backward_diff = backward_theta_pred-theta_r[0:-1,:]

		lossf = tf.reduce_mean(tf.reduce_sum(f_pred**2,axis=1))
		lossdiff = tf.reduce_sum(tf.reduce_sum(backward_diff**2,axis=1))
		if log:
			self.loss_log[2].append(lossf.numpy())
			self.loss_log[9].append(lossdiff.numpy())
		loss = lossf+lossdiff
		return loss

	def loss_boundary_data(self, bound, toggle, log = False):
		psi_pred, K_pred, theta_pred, f_pred, flux_pred, [psiz_pred, psit_pred, thetat_pred, Kz_pred, psizz_pred, flux_residual] = self.rre_model(bound['z'], bound['t'], bound['flux'])
		_,_,bbweight,tbweight = self.weight_scheduling()
		weight = bbweight if toggle == 'bottom' else tbweight
		if bound['type'] == 'flux':
			loss = self.loss_f(flux_residual)
		elif bound['type'] == 'psiz':
			loss = self.loss_reduce_mean(psiz_pred, bound['data'])
		elif bound['type'] == 'psi':
			loss = self.loss_reduce_mean(psi_pred, bound['data'])/(self.psi_ub-self.psi_lb)**2
		if log:
			if toggle == 'top':
				self.loss_log[7].append(weight.numpy())
				self.loss_log[3].append(loss.numpy())
			elif toggle == 'bottom':
				self.loss_log[8].append(weight.numpy())
				self.loss_log[4].append(loss.numpy())
		return loss*weight

	def predict(self, z_tf, t_tf, flux_tf):
		z_tf = self.convert_tensor(z_tf)
		t_tf = self.convert_tensor(t_tf)
		flux_tf = self.convert_tensor(flux_tf)
		psi, K, theta = self.rre_model_pred(z_tf, t_tf, flux_tf)
		return psi.numpy(), K.numpy(), theta.numpy()

	def convert_bound_data(self, bound):
		bound['z'] = self.convert_tensor(bound['z'])
		bound['t'] = self.convert_tensor(bound['t'])
		bound['flux'] = self.convert_tensor(bound['flux'])
		bound['data'] = self.convert_tensor(bound['data'])
		return bound

	def convert_residual_data(self, residual):
		residual['z'] = self.convert_tensor(residual['z'])
		residual['t'] = self.convert_tensor(residual['t'])
		residual['flux'] = self.convert_tensor(residual['flux'])
		return residual

