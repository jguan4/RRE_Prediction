from RRENetwork import RRENetwork
from RRENetwork_flux import RRENetwork_flux
from RRETestProblem import RRETestProblem
import pandas as pd
import numpy as np
import os
from tkinter import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from utils import *


def main_loop(psistruct, Kstruct, thetastruct, training_hp, test_hp, train_toggle):
	train_env = RRETestProblem(training_hp)
	pathname = train_env.DataFolder

	training_hp['lb'] = train_env.lb
	training_hp['ub'] = train_env.ub

	training_hp['psi_lb'] = train_env.env.psi_lb
	training_hp['psi_ub'] = train_env.env.psi_ub

	theta_data, residual_data, boundary_data = train_env.get_training_data()


	if training_hp['network_toggle'] == 'RRENetwork':
		rrenet = RRENetwork(psistruct, Kstruct, thetastruct, training_hp, pathname)
	elif training_hp['network_toggle'] == 'RRENetwork_flux':
		rrenet = RRENetwork_flux(psistruct, Kstruct, thetastruct, training_hp, pathname)

	if train_toggle == 'train':
		rrenet.fit(theta_data, residual_data,boundary_data)
		end_epoch = rrenet.epoch

	elif train_toggle == 'retrain':
		rrenet.load_model()
		rrenet.fit(theta_data, residual_data,boundary_data)
		end_epoch = rrenet.epoch

	elif train_toggle == 'test':
		rrenet.load_model()
		test_data = train_env.get_testing_data()

		if training_hp['network_toggle'] == 'RRENetwork':
			ttest, ztest, fluxtest, psitest, thetatest, Ktest = test_data
			psi_pred, K_pred, theta_pred = rrenet.predict(ztest,ttest,fluxtest)
		elif training_hp['network_toggle'] == 'RRENetwork_flux': 
			ttest, ztest, psitest, thetatest, Ktest = test_data
			psi_pred, K_pred, theta_pred = rrenet.predict(ztest,ttest)

		order_str = ['theta','K','psi']

		for (ostr,data, pred) in zip(order_str,[thetatest, Ktest, psitest], [theta_pred,K_pred,psi_pred]):
			err = relative_error(data,pred)
			print("For {0}, relative error is {1}.\n".format(ostr,err))

	return end_epoch


train_toggle = 'train' # choose the mode, available options include 'train' (training from scrach), 'retrain' (restart traing from exisiting weights), 'test' (test the network and show relative errors)

starting_epoch = 0 # starting_epoch: the number of epoch the training process wants to start at, 0 for training from scratch, change to the last epoch number if you are retraining based on existing weights

name = 'Test_Plot' # problem name, available options: 'Test1' (Celia 1990 test problem), 'Bandai_SandyLoam' (Bandai 2021 test problem), 'Test_Plot' (CRREL/C5ISR test plot data) 

# training_period defines parameters needed to generate training samples, need the following parameters:
# - start_t: start of the time window desired
# - end_t: end of the time window desired
# - measured_start: start of the time theta samples are provided
# - measured_end: end of the time theta samples are provided
# - theta_depths: depths where theta samples are provided
training_period = {'start_t':57, 'end_t':60, 'measured_start':57, 'measured_end':58.5, 'theta_depths':np.array([-15,-35,-55])}

# LBFGS training parameters: no need to alter. But if you want train for more epochs, can change 'maxfun' and 'maxiter' to the desired epoch number
lbfgs_options={'disp': None, 'maxcor': 50, 'ftol': 2.220446049250313e-16, 'gtol': 1e-05, 'maxfun': 50000, 'maxiter': 50000, 'maxls': 50, 'iprint':1}
# Adam training parameters: no need to alter. But if you want train for more epochs, can change 'epoch'
adam_options = {'epoch':150000}
total_epoch = lbfgs_options['maxiter'] + adam_options['epoch']

# training_hp defines parameters needed for training, need the following parameters:
# - name: problem name, available options: Test1 (Celia 1990 test problem), SandyLoam (Bandai 2021 test problem), TestPlot (CRREL/C5ISR test plot data) 
# - dz: space step size
# - dt: time step size 
# - Z: deepest depth
# - T: final time
# - noise: noise level added to theta samples
# - lbfgs_options: training parameters for LBFGS, no need to modify unless want to change training process
# - adam_options: training parameters for Adam, no need to modify unless want to change training process
# - norm: choose how to normalize inputs, '_norm' for normailizing to [-1,1], '_norm1' for normalizing to [0,1]
# - weights: weighting of MSEs, it is in the form of [bottom boundary weight, residual weight, theta weight, top boundary weight]
# - starting_epoch: the number of epoch the training process wants to start at, 0 for training from scratch, change to the last epoch number if you are retraining based on existing weights
# - scheduling_toggle: choose how to schedule weights of the MSEs, available options include 'constant' (do not change throughout training), 'linear' (linearly increase, specific parameters can be altered in RRENetwork.py), 'exp' (exponentially increase, specific parameters can be altered in RRENetwork.py)
# - network_toggle: choose the network structure to use, available options include 'RRENetwork' ((z,t) as inputs), 'RRENetwork_flux' ((z,t,flux) as inputs)
# - training_period: defined above
training_hp = {'name':name, 'dz': 0.1, 'dt': 0.012, 'Z':65, 'T':3, 'noise':0, 'lbfgs_options':lbfgs_options, 'adam_options':adam_options, 'norm':'_norm', 'weights': [1, 1e-3, 1e3, 1], 'starting_epoch': starting_epoch, 'total_epoch':total_epoch, 'scheduling_toggle':'linear', 'network_toggle':'RRENetwork_flux', 'training_period':training_period}

# training_hp defined parameters needed for testing, need the following parameters:
# - name: problem name, available options: Test1 (Celia 1990 test problem), SandyLoam (Bandai 2021 test problem), TestPlot (CRREL/C5ISR test plot data) 
# - dz: space step size
# - dt: time step size 
# - Z: deepest depth
# - T: final time
test_hp = {'name':'Test1', 'dz': 0.1, 'dt': .012,'Z':100, 'T':3, 'noise':0}

# Network Structures (Optimal ones according to Bandai 2021)
input_num = 2 if training_hp['network_toggle'] == 'RRENetwork' else 3
psistruct = {'layers':[input_num,40,40,40,40,40,40,1],'toggle':'DNN'} 
Kstruct = {'layers':[1,40,40,40,1],'toggle':'MNN'} 
thetastruct = {'layers':[1,40,1],'toggle':'MNN'} 

end_epoch = main_loop(psistruct, Kstruct, thetastruct, training_hp, test_hp, train_toggle)
print("Training has ended. The ending epoch is {0}.\n".format(end_epoch))
