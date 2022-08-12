import numpy as np
import scipy
import pandas as pd
from utils import *

# Test problem 2: from Bandai 2021
# Bottom Dirchlet pressure head boundary, top flux boundary
# dz = 0.1, dt = 0.012, Z = -100, T = 3 from hydrus-1d simulation
class RRE_SandyLoam(object):
	def __init__(self, training_hp):
		self.name = "Bandai_SandyLoam"
		self.training_hp = training_hp

		self.dt = 0.012 
		self.dz = 0.1
		self.T = 3
		self.Z = -100 

		self.Nt = int(self.T/self.dt)+1
		self.Nz = int((-self.Z/self.dz))+1
		self.N = self.Nt*self.Nz

		self.lb = [-100,0]
		self.ub = [0,3]

		self.psi_lb = -1000.0
		self.psi_ub = -12.0

	def get_data(self, train_toggle = 'training'):
		training_period = self.training_hp['training_period']
		start_t = training_period['start_t']
		end_t = training_period['end_t']
		theta_depths = training_period['theta_depths']
		measured_start = training_period['measured_start']
		measured_end = training_period['measured_end']

		start_t_ind = int(start_t/self.dt)
		end_t_ind = int(end_t/self.dt)
		theta_depths_inds = (-theta_depths/self.dz).astype('int')
		measured_start_ind = int(measured_start/self.dt)
		measured_end_ind = int(measured_end/self.dt)

		data = pd.read_csv('./sandy_loam_nod.csv')
		t = data['time'].values[:,None]
		z = data['depth'].values[:,None]
		psi = data['head'].values[:,None]
		K = data['K'].values[:,None]
		C = data['C'].values[:,None]
		theta = data['theta'].values[:,None]
		flux = data['flux'].values[:,None]

		zt, tt, thetat = extract_data_time_space_inds([z,t,theta], Nt = self.Nt, Nz = self.Nz, spaceinds=theta_depths_inds, Nst = measured_start_ind, Net = measured_end_ind, Nit = 1)

		original_times = np.linspace(0,3,self.Nt)
		zff =  np.linspace(-100, 0, 101)
		tfff = original_times
		Z,T = np.meshgrid(zff,original_times)
		zf = np.reshape(Z,[np.prod(Z.shape),1])
		tf = np.reshape(T,[np.prod(T.shape),1])

		ztb, ttb, fluxtb = extract_top_boundary([z,t,flux])
		zbb, tbb, psibb = extract_bottom_boundary([z,t,psi], Nb = self.Nz-1)

		flux_inputs = []
		for item in [tt,tf,ttb,tbb]:
			flux_inputs.append(flux_function(item))

		fluxt, fluxf, fluxtb, fluxbb = flux_inputs
			
		theta_data = {'z':zt, 't':tt, 'flux':fluxt, 'data':thetat}
		residual_data = {'z':zf, 't':tf, 'flux':fluxf, 'nz':101}
		boundary_data = {'top':{'z':ztb, 't':ttb, 'flux':fluxtb, 'data':fluxtb, 'type':'flux'}, 'bottom':{'z':zbb, 't':tbb, 'flux':fluxbb, 'data':psibb, 'type':'psi'}}

		if train_toggle == 'training':
			return theta_data, residual_data, boundary_data
		elif train_toggle == 'testing':
			return t,z,flux,psi,theta,K


		
