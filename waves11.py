from __future__ import division
# Georgia State University
# Neuroscience Institue
# Ricardo Erazo
# traveling waves in HH endogenous oscillatory neurons.

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint


nr_neurons = 20
ap1 = nr_neurons+1
tau2 = 3
delta1 = 1e-2
sigma1 = 3e-2
gsyn= 20
domain = np.arange(0,nr_neurons)

Vt = 0.02 
Ie = 0.1
# Isyn = Ie*np.exp(-np.arange(0,nr_neurons,1)*delta1/sigma1)
Ispike = Ie*np.exp(-np.arange(0,nr_neurons,1)*delta1/sigma1)

#Experimental parameters
Htheta= 0.041326
K2theta= -0.0075

#Cell parameters
gNa= 105
ENa= 0.045
gK2= 30
EK=-0.07
gH= 4
EH= -0.021
gl= 8
El= -0.046
Vrev= 0.01

#time scales
Cm= 0.5
tau_h= 0.04050
tau_m= 0.1
tau_k= 2
tau_s= 0.1
tau1= Cm;
#
# pars = [tau2, Ie2, delta1, sigma1, gsyn, Htheta, K2theta, gNa, ENa, gK2, EK, gH, EH, gl, El, Vrev, Cm, tau_h, tau_k, tau_s, tau1, tau_m]
# ODE
#initial conditions
## rest
v0= -0.04220452
h0= 0.99225122
m0= 0.29849783
n0= 0.01681514
## shocked
v2 = -0.036892644544022
h2 = 0.867431828696752
m2 = 0.016203233894593
n2 = 0.175309528389848

#First shocked neuron
def vecf(t, y0):
	v,h,m,n = y0
	mNass=1./(1.+np.exp(-150.*(y0[0]+0.0305)))
	mK2ss=1./(1.+np.exp(-83.*(y0[0]+K2theta)))
	Ie_time_dependent = Ie_local*np.exp(-t/tau2)

	dv = (-1/Cm) * (gNa*mNass*mNass*mNass*y0[1]*(y0[0]-ENa)+ gK2*y0[3]*y0[3]*(y0[0]-EK) + gH*y0[2]*y0[2]*(y0[0]-EH) + gl*(y0[0]-El) + 0.006 +gsyn*Ie_time_dependent*(y0[0]-Vrev))
	dh = (1/(1+ np.exp(500*(y0[0]+0.0325))) - y0[1])/tau_h
	dm = (1/(1+2*np.exp(180*(y0[0]+Htheta))+np.exp(500*(y0[0]+Htheta))) -y0[2])/tau_m
	dn = (mK2ss - y0[3])/tau_k
	f = dv, dh, dm, dn
	return f

def Vt_cross(t,y0): return y0[0]-Vt
Vt_cross.terminal= True
Vt_cross.direction = 1

#initial conditions
y0 = np.array((v0,h0,m0,n2))
y2 = np.array((v2,h2,m2,n2))
spike_times = np.zeros((1,nr_neurons))
statevar = np.zeros((nr_neurons,4))
for i in np.arange(0,nr_neurons):
	statevar[i,:] = y0
t1= 0
tspan = [0, 10]
Ie_local = Ispike[0]
pre_n = solve_ivp(vecf, tspan, y0,events= Vt_cross, method = 'RK45',rtol=1e-5,atol=1e-7)
tspan = [t1, pre_n.t_events[0]]
for i in np.arange(0,nr_neurons):
	# print i
	Ie_local = Ispike[i]
	neur1 = solve_ivp(vecf, tspan, statevar[i,:], events= Vt_cross, method = 'RK45',rtol=1e-5,atol=1e-7)
	statevar[i,:] = neur1.y[:,-1]
	if np.size(neur1.t_events[0]):
		spike_times[[0],[i]] = neur1.t_events[0]
	plt.ion()
	plt.plot(neur1.t, neur1.y[0,:])
plt.plot(pre_n.t,pre_n.y[0,:])