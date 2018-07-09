from __future__ import division
# Georgia State University
# Neuroscience Institue
# Ricardo Erazo
# traveling waves in HH endogenous oscillatory neurons.

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


nr_neurons = np.int(1)
tau2 = np.int(3)
Ie2 = np.float(0.01)
delta1 = np.float(1e-2)
sigma1 = np.float(3e-2)
gsyn= np.int(20)
domain = np.arange(1,nr_neurons+1)
# print domain
Isyn = Ie2*np.exp(-(np.transpose(domain)*delta1/sigma1))
# temp1 = domain*delta1/sigma1
Isyn = Ie2*np.exp(-(domain*delta1/sigma1))
# print Isyn
Vt = np.float(0.02)

#Experimental parameters
Htheta= np.float(0.041326)
K2theta= np.float(-0.0075)

#Cell parameters
gNa= np.float(105)
ENa= np.float(0.045)
gK2= np.float(30)
EK=np.float(-0.07)
gH= np.float(4)
EH= np.float(-0.021)
gl= np.float(8)
El= np.float(-0.046)
Vrev= np.float(0.01)

#time scales
Cm= np.float(0.5)
tau_h= np.float(0.04050)
tau_m= np.float(0.1)
tau_k= np.float(2)
tau_s= np.float(0.1)
tau1= Cm;
#
pars = [tau2, Ie2, delta1, sigma1, gsyn, Isyn, Htheta, K2theta, gNa, ENa, gK2, EK, gH, EH, gl, El, Vrev, Cm, tau_h, tau_k, tau_s, tau1, tau_m]
# ODE
#initial conditions

yy0 = [0]*nr_neurons*4
for i in range(0, nr_neurons):
	# print i
	yy0[i] = np.float(-0.042391755919600)
	yy0[i+nr_neurons*1] = np.float(0.992950238576496)
	yy0[i+nr_neurons*2] = np.float(0.022726154481198)
	yy0[i+nr_neurons*3] = np.float(0.309985836203579)
	# yy0[i] = np.float(-0.036892644544022)
	# yy0[i+nr_neurons*1] = np.float(0.867431828696752)
	# yy0[i+nr_neurons*2] = np.float(0.016203233894593)
	# yy0[i+nr_neurons*3] = np.float(0.175309528389848)

y0= np.zeros((1,4))

y0 = [0] * 1 * 4
## rest
y0[0]= -0.042391755919600
y0[1]=0.992950238576496
y0[2]=0.022726154481198
y0[3]=0.309985836203579
## shocked
# y0[0] = -0.036892644544022
# y0[1] = 0.867431828696752
# y0[2] = 0.016203233894593
# y0[3] = 0.175309528389848

dt = np.float(0.0001)
t1= 0
tend = 150
#Simple neuron model
def vecf(t, y0):
	v = y0[0]
	h = y0[1]
	m = y0[2]
	n = y0[3]
	mNass=1./(1.+np.exp(-150.*(y0[0]+0.0305)))
	mK2ss=1./(1.+np.exp(-83.*(y0[0]+K2theta)))
	Ie_time_dependent = Isyn*np.exp(-t/tau2)

	v = (-1/Cm) * (gNa*mNass*mNass*mNass*y0[1]*(y0[0]-ENa)+ gK2*y0[3]*y0[3]*(y0[0]-EK) + gH*y0[2]*y0[2]*(y0[0]-EH) + gl*(y0[0]-El) + 0.006)
	h = (1/(1+ np.exp(500*(y0[0]+0.0325))) - y0[1])/tau_h
	m = (1/(1+2*np.exp(180*(y0[0]+Htheta))+np.exp(500*(y0[0]+Htheta))) -y0[2])/tau_m
	n = (mK2ss - y0[3])/tau_k
	f = v, h, m, n
	return f

xsol = solve_ivp(vecf, [t1, tend], y0, method='RK45',rtol=1e-6,atol=1e-6)#,max_step=1e-3)

plt.ion()
plt.plot(xsol.t, xsol.y[0,:],label='rest')

#Single neuron. Synaptic current added
def vecf(t, y0):
	v = y0[0]
	h = y0[1]
	m = y0[2]
	n = y0[3]
	mNass=1./(1.+np.exp(-150.*(y0[0]+0.0305)))
	mK2ss=1./(1.+np.exp(-83.*(y0[0]+K2theta)))
	Ie_time_dependent = Ie2*np.exp(-t/tau2)
	v = (-1/Cm) * (gNa*np.power(mNass,3)*y0[1]*(y0[0]-ENa)+ gK2*np.power(y0[3],2)*(y0[0]-EK) + gH*np.power(y0[2],2)*(y0[0]-EH) + gl*(y0[0]-El) + 0.006 - Ie_time_dependent)
	h = (1/(1+ np.exp(500*(y0[0]+0.0325))) - y0[1])/tau_h
	m = (1/(1+2*np.exp(180*(y0[0]+Htheta))+np.exp(500*(y0[0]+Htheta))) -y0[2])/tau_m
	n = (mK2ss - y0[3])/tau_k
	f = v, h, m, n
	return f

sol = solve_ivp(vecf, [t1, tend], y0, method='RK45',rtol=1e-5,atol=1e-7)#,max_step=1e-3)
plt.plot(sol.t, sol.y[0,:],label='shocked')
plt.legend()


def vecf_par(t, y0):
	v = np.zeros((1,nr_neurons))
	h = np.zeros((1,nr_neurons))
	m = np.zeros((1,nr_neurons))
	n = np.zeros((1,nr_neurons))
	v,h,m,n = y0
	mNass=np.true_divide(1,(1+np.exp(-150.*(v+0.0305))))
	mK2ss=np.true_divide(1,(1+np.exp(-83.*(v+K2theta))))
	Ie_time_dependent = Ie2*np.exp(-t/tau2)

	dv = (-1/Cm) * (gNa*np.dot(np.power(mNass,3),np.dot(h,(v-ENa)))+ gK2*np.dot(np.power(n,2),(v-EK)) + gH*np.dot(np.power(m,2),(v-EH)) + gl*(v-El) + 0.006 - Ie_time_dependent)
	dh = np.true_divide((np.true_divide(1,(1+ np.exp(500*(v+0.0325)))) - h),tau_h)
	dm = np.true_divide((np.true_divide(1,(1+2*np.exp(180*(v+Htheta))+np.exp(500*(v+Htheta)))) -m),tau_m)
	dn = np.true_divide((mK2ss - n),tau_k)
	f = dv, dh, dm, dn
	return f

sol = solve_ivp(vecf_par, [t1+10, t1+11], yy0, method='RK45',rtol=1e-5,atol=1e-7)#,max_step=1e-3)
plt.plot(sol.t, sol.y[0,:],label='one')
plt.legend()

