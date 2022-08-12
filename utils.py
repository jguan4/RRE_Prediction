import numpy as np
import pandas as pd


def extract_data(data, Nt = 251, Nz = 1001, Ns = 10, Ne = 200, Ni = 20):
	train_data = []
	for item in data:
		Item = np.reshape(item,[Nt,Nz])
		Items = Item[:,Ns:Ne:Ni]
		Itemt = np.reshape(Items,[np.prod(Items.shape),1])
		train_data.append(Itemt)
	return train_data

def extract_data_timewise(data, Nt = 251, Nz = 1001, Ns = 10, Ne = 200, Ni = 20):
	train_data = []
	for item in data:
		Item = np.reshape(item,[Nt,Nz])
		Items = Item[Ns:Ne:Ni,:]
		Itemt = np.reshape(Items,[np.prod(Items.shape),1])
		train_data.append(Itemt)
	return train_data

def extract_data_time_space(data, Nt = 251, Nz = 1001, Ns = 10, Ne = 200, Ni = 20, Nst = 0, Net = 251, Nit = 1):
	train_data = []
	for item in data:
		Item = np.reshape(item,[Nt,Nz])
		Items = Item[Nst:Net:Nit,Ns:Ne:Ni]
		Itemt = np.reshape(Items,[np.prod(Items.shape),1])
		train_data.append(Itemt)
	return train_data

def extract_data_time_space_inds(data, Nt = 251, Nz = 1001, spaceinds=range(10,200,10), Nst = 0, Net = 251, Nit = 1):
	train_data = []
	for item in data:
		Item = np.reshape(item,[Nt,Nz])
		Items = Item[Nst:Net:Nit,spaceinds]
		Itemt = np.reshape(Items,[np.prod(Items.shape),1])
		train_data.append(Itemt)
	return train_data

def flux_function(t, toggle = 'Bandai1'):
	flux = np.zeros(t.shape)
	if toggle == 'Bandai1':
		flux[np.argwhere(np.logical_and(t>0.0,t<0.25))[:,0]] = -10
		flux[np.argwhere(np.logical_and(t>=0.5,t<1.0))[:,0]] = 0.3
		flux[np.argwhere(np.logical_and(t>=1.5,t<2.0))[:,0]] = 0.3
		flux[np.argwhere(np.logical_and(t>=2.0,t<2.25))[:,0]] = -10
		flux[np.argwhere(np.logical_and(t>=2.5,t<=3.0))[:,0]] = 0.3
	elif 'test_plot' in toggle:
		name = toggle.split('.')[0]
		subfix = name.replace('test_plot_data','')
		data = pd.read_csv('./'+toggle)
		times = data['time'].values[:,None]
		fluxs = data['flux'].values[:,None]

		maxind = int(np.squeeze(np.argwhere(np.max(t)<=times)))
		minind = int(np.squeeze(np.argwhere(np.min(t)>=times)))
		for i in range(minind,maxind):
			if i == 0:
				ind = 0
			else:
				ind = np.argwhere(np.logical_and(t<=times[i], t>times[i-1]))
			flux[ind] = fluxs[i]
	return flux

def extract_top_boundary(data, Nt = 251, Nz = 1001):
	train_data = []
	for item in data:
		Item = np.reshape(item,[Nt,Nz])
		Items = Item[:,[0]]
		Itemt = np.reshape(Items,[np.prod(Items.shape),1])
		train_data.append(Itemt)
	return train_data

def extract_bottom_boundary(data,Nt = 251, Nz = 1001, Nb = None):
	train_data = []
	for item in data:
		Item = np.reshape(item,[Nt,Nz])
		Items = Item[:,[-1]] if Nb is None else Item[:,[Nb]]
		Itemt = np.reshape(Items,[np.prod(Items.shape),1])
		train_data.append(Itemt)
	return train_data

def extract_data_test(data, Nz = 1001, Nt = 251, dt = 0.012, T = None, Ni = 20):
	test_data = []
	for item in data:
		Item = np.reshape(item,[Nt,Nz])
		if T is None:
		# Items = Item[int(T/0.012),0:200]
			Items = Item[:,::Ni]
		else:
			Items = Item[int(T/dt),::Ni]
		Itemt = np.reshape(Items,[np.prod(Items.shape),1])
		test_data.append(Itemt)
	return test_data

def relative_error(data, pred):
	# data = np.reshape(data,[251,200])
	# pred = np.reshape(pred,[251,200])
	error = np.sum((data-pred)**2,axis = None)/np.sum(data**2,axis = None)
	return error

def psi_func(theta):
	thetas = 0.41
	thetar = 0.065
	n = 1.89
	m = 1-1/n
	alpha = 0.075
	Psi = (((thetas-thetar)/(theta-thetar))**(1/m)-1)**(1/n)/(-alpha)
	return Psi

def load_csv_data(csv_file):
	if csv_file == 'sandy_loam_nod.csv':
		data = pd.read_csv('./'+csv_file)
		t = data['time'].values[:,None]
		z = data['depth'].values[:,None]
		psi = data['head'].values[:,None]
		K = data['K'].values[:,None]
		C = data['C'].values[:,None]
		theta = data['theta'].values[:,None]
		flux = data['flux'].values[:,None]
		T = None
		return [t,z,psi,K,C,theta,flux]
	elif 'test_plot' in csv_file:
		name = self.training_hp['csv_file'].split('.')[0]
		subfix = name.replace('test_plot_data','')
		data = pd.read_csv('./'+csv_file)
		
		t = data['time'].values[:,None]
		z = data['depth'].values[:,None]
		flux = data['flux'].values[:,None]
		theta = data['theta'].values[:,None]
		T = t[int(Nt/4)]

		tbdata = pd.read_csv('./test_plot_tb'+subfix+'.csv')
		ttb = tbdata['time'].values[:,None]
		ztb = tbdata['depth'].values[:,None]
		fluxtb = tbdata['flux'].values[:,None]
		T = None
		return [t,z,flux,theta,ttb,ztb,fluxtb]	