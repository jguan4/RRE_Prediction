import numpy as np
import os
from RRE_test1 import RRE_test1
from RRE_SandyLoam import RRE_SandyLoam
from RRE_TestPlot import RRE_TestPlot

class RRETestProblem(object):
	def __init__(self, training_hp):
		self.training_hp = training_hp
		self.name = self.training_hp['name']
		if self.name == 'Test1':
			self.env = RRE_test1(self.training_hp)
		elif self.name == 'Bandai_SandyLoam':
			self.env = RRE_SandyLoam(self.training_hp)
		elif self.name == 'Test_Plot':
			self.env = RRE_TestPlot(self.training_hp)
		if self.training_hp['network_toggle'] == 'RRENetwork':	
			self.lb = self.env.lb
			self.ub = self.env.ub
		elif self.training_hp['network_toggle'] == 'RRENetwork_flux':	
			self.lb = self.env.lb+[0]
			self.ub = self.env.ub+[1]
		self.define_folder_names()

	def get_training_data(self):
		self.generate_folder()
		print("calling {0}...\n".format(self.name))
		theta_data, residual_data, boundary_data = self.env.get_data(train_toggle = 'training')
		Thetas = theta_data['data']
		# standard normal distibuiton with 0 mean and stndard error of the noise value
		noise_theta = self.training_hp['noise']*np.random.randn(Thetas.shape[0], Thetas.shape[1]) 
		Thetas = Thetas + noise_theta
		theta_data['data'] = Thetas
		return theta_data, residual_data, boundary_data

	def get_testing_data(self):
		print("calling {0}...\n".format(self.name))
		test_data = self.env.get_data(train_toggle = 'testing')
		return test_data

	def define_folder_names(self):
		self.EnvFolder = './'+self.training_hp['name']+'_checkpoints'
		weights = self.training_hp['weights']
		add_tag = "_theta{1}_f{0}_tb{2}_lb{3}".format(weights[1], weights[2], weights[3],weights[0])
		self.DataFolder = self.EnvFolder+"/{5}_Nt{0}_Nz{1}_noise{2}{3}{4}".format(self.env.Nt,self.env.Nz,self.training_hp['noise'],self.training_hp['norm'],add_tag,self.training_hp['network_toggle'])

	def generate_folder(self):
		if not os.path.exists(self.EnvFolder):
			os.makedirs(self.EnvFolder)
		if not os.path.exists(self.DataFolder):
			os.makedirs(self.DataFolder)