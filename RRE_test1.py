import numpy as np
import scipy
import scipy.sparse.linalg

# Test problem 1: from Celia 1990
# Both Dirchlet boundaries, relatively smooth
class RRE_test1(object):
	def __init__(self,training_hp):
		self.name = "Test1"

		self.res_tol = 1e-5
		self.dt = training_hp['dt'] 
		self.dz = training_hp['dz'] 
		self.T = training_hp['T']
		self.Z = training_hp['Z'] 

		self.Nt = int(self.T/self.dt)+1
		self.Nz = int(self.Z/self.dz)+1
		self.N = self.Nt*self.Nz
		
		self.zs = np.linspace(0, self.Z, self.Nz)
		self.zs = np.reshape(self.zs, [self.Nz, 1])
		self.ts = np.linspace(0,self.T, self.Nt)

		self.lb = [0,0]
		self.ub = [40,360]

		self.psi_lb = -61.5
		self.psi_ub = -20.7

	def get_data(self, train_toggle = 'training'):
		e = np.ones([self.Nz,1])
		H = self.init_condition()
		H = self.boundary_condition(H)

		Theta = self.theta(H)
		Theta_old = Theta

		C = self.c(H)
		K = self.k(H)

		if train_toggle == 'training':
			ts = np.array([])
			zs = np.array([])
			Hs = np.array([])
			Thetas = np.array([])
			Ks = np.array([])

		for nt in range(1, self.Nt+1):
			t = self.dt*nt
	
			resi = self.residual(H,K,Theta,Theta_old)
			res = np.linalg.norm(resi)

			count = 0
			while res>self.res_tol:
				count += 1
				A = self.build_mat(K,C)
				dH = scipy.sparse.linalg.spsolve(A,resi)
				H = H+np.reshape(dH,H.shape)

				Theta = self.theta(H)
				C = self.c(H)
				K = self.k(H)
				resi = self.residual(H,K,Theta,Theta_old)
				res = np.linalg.norm(resi)
			print("Inner loop converge at iteration {0} with residual {1}\n".format(count, res))

			if train_toggle == 'training':
				ts = np.vstack([ts, t*e]) if ts.size else t*e
				zs = np.vstack([zs, self.zs]) if zs.size else self.zs
				Hs = np.vstack([Hs, H]) if Hs.size else H
				Thetas = np.vstack([Thetas, Theta]) if Thetas.size else Theta
				Ks = np.vstack([Ks, K]) if Ks.size else K

			Theta_old = Theta
		zt = np.reshape(zs,[np.prod(zs.shape),1])
		tt = np.reshape(ts,[np.prod(ts.shape),1]) 
		thetat = np.reshape(Thetas,[np.prod(Thetas.shape),1]) 
		psit = np.reshape(Hs,[np.prod(Hs.shape),1]) 
		Kt = np.reshape(Ks,[np.prod(Ks.shape),1]) 

		tbb = np.reshape(self.ts,[len(self.ts),1])
		zbb = self.lb[0]*np.ones(tbb.shape)
		psibb = -61.5*np.ones(tbb.shape)

		ttb = np.reshape(self.ts,[len(self.ts),1])
		ztb = self.ub[0]*np.ones(ttb.shape)
		psitb = -20.7*np.ones(ttb.shape)
		
		theta_data = {'z':zt, 't':tt, 'data':thetat}
		residual_data = {'z':zt, 't':tt}
		boundary_data = {'top':{'z':ztb, 't':ttb, 'data':psitb, 'type':'psi'}, 'bottom':{'z':zbb, 't':tbb, 'data':psibb, 'type':'psi'}}
		if train_toggle == 'training':
			return theta_data, residual_data, boundary_data
		else:
			return tt, zt, psit, thetat, Kt

	def boundary_condition(self, H):
		H[0] = -61.5
		H[-1] = -20.7
		return H

	def init_condition(self):
		H = -61.5*np.ones([self.Nz,1])
		return H

	def residual(self,H,K,Theta,Theta_old):
		Kh = (K[0:-1]+K[1::])/2
		p1 = (Kh[1::]*(H[2::]-H[1:-1])-Kh[0:-1]*(H[1:-1]-H[0:-2]))/self.dz**2
		p2 = (Kh[1::]-Kh[0:-1])/self.dz
		p3 = (Theta_old-Theta)/self.dt
		resi = np.zeros((self.Nz,1))
		resi[1:-1] = p1+p2+p3[1:-1]
		return resi

	def build_mat(self, K,C):
		Kh = (K[0:-1]+K[1::])/2
		bm = C[1:-1]/self.dt+(Kh[0:-1]+Kh[1::])/self.dz**2
		a = np.append(-Kh[0:-1]/self.dz**2,[0,0])
		c = np.append([0,0],-Kh[1::]/self.dz**2)
		b = np.append(np.append([1],bm),[1])
		A = scipy.sparse.spdiags([a,b,c],[-1,0,1],self.Nz,self.Nz)
		return A

	def theta(self, H):
		thetas = 0.287
		thetar = 0.075
		beta = 3.96
		alpha = 1.611e6
		Theta = (alpha*(thetas-thetar))/(alpha+(-H)**beta)+thetar
		return Theta

	def c(self, H):
		thetas = 0.287
		thetar = 0.075
		beta = 3.96
		alpha = 1.611e6
		C = (alpha*(thetas-thetar)*beta*(-H)**(beta-1))/(alpha+(-H)**beta)**2
		return C

	def k(self, psi):
		Ks = 0.00944
		A = 1.75e6
		gamma = 4.74
		K = Ks*A/(A+(-psi)**gamma)
		return K