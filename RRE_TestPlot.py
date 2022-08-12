import numpy as np
import scipy
import pandas as pd
from utils import *

# Test problem 2: from Bandai 2021
# Bottom Dirchlet pressure head boundary, top flux boundary
class RRE_TestPlot(object):
	def __init__(self, training_hp):
		self.name = 'Test_Plot'
		self.training_hp = training_hp

		self.dt = 15/(24*60)
		self.dz = 20
		self.T = 198
		self.Z = -65

		self.Nt = 19006
		self.Nz = 3
		self.N = self.Nt*self.Nz

		self.lb = [-65,0]
		self.ub = [0,198]

		self.psi_lb = -280
		self.psi_ub = 0
		

	def get_data(self, train_toggle = 'training'):
		training_period = self.training_hp['training_period']
		start_t = training_period['start_t']
		end_t = training_period['end_t']
		theta_depths = training_period['theta_depths']
		measured_start = training_period['measured_start']
		measured_end = training_period['measured_end']

		start_t_ind = int(start_t/self.dt)
		end_t_ind = int(end_t/self.dt)
		measured_start_ind = int(measured_start/self.dt)
		measured_end_ind = int(measured_end/self.dt)

		csv_file = "test_plot_data_1.csv" 
		name = csv_file.split('.')[0]
		subfix = name.replace('test_plot_data','')
		data = pd.read_csv('./'+csv_file)

		t = data['time'].values[:,None]
		z = data['depth'].values[:,None]
		flux = data['flux'].values[:,None]
		theta = data['theta'].values[:,None]
		zt, tt, fluxt, thetat = extract_data_timewise([z,t,flux,theta],Nt = self.Nt, Nz = self.Nz, Ns = measured_start_ind, Ne = measured_end_ind, Ni = 1)

		tbdata = pd.read_csv('./test_plot_tb'+subfix+'.csv')
		ttb = tbdata['time'].values[:,None]
		ztb = tbdata['depth'].values[:,None]
		fluxtb = tbdata['flux'].values[:,None]
		ztb, ttb, fluxtb = extract_data_timewise([ztb,ttb,fluxtb],Nt = self.Nt, Nz = 1, Ns = measured_start_ind, Ne = measured_end_ind, Ni = 1)

		bbdata = pd.read_csv('./test_plot_bb'+subfix+'.csv')
		tbb = bbdata['time'].values[:,None]
		zbb = bbdata['depth'].values[:,None]
		fluxbb = bbdata['flux'].values[:,None]
		psibb = bbdata['psi'].values[:,None]
		zbb, tbb, fluxbb, psibb = extract_data_timewise([zbb,tbb,fluxbb,psibb],Nt = self.Nt, Nz = 1, Ns = measured_start_ind, Ne = measured_end_ind, Ni = 1)

		zs = np.linspace(-65,0,27)
		zs = zs
		ts = ttb 
		fluxs = fluxtb 
		Z,T = np.meshgrid(zs,ts)
		Fluxs = np.tile(np.reshape(fluxs,[len(fluxs),1]),(1,len(zs)))
		zf = Z.flatten()
		tf = T.flatten()
		fluxf = Fluxs.flatten()

		theta_data = {'z':zt, 't':tt, 'flux':fluxt, 'data':thetat}
		residual_data = {'z':zf, 't':tf, 'flux':fluxf,'nz':27}
		boundary_data = {'top':{'z':ztb, 't':ttb, 'flux':fluxtb, 'data':fluxtb, 'type':'flux'}, 'bottom':{'z':zbb, 't':tbb, 'flux':fluxbb, 'data':psibb, 'type':'psi'}}

		if train_toggle == 'training':
			return theta_data, residual_data, boundary_data
		elif train_toggle == 'testing':
			return t,z,flux,None,theta,None