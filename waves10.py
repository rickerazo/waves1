from __future__ import division
# Georgia State University
# Neuroscience Institue
# Ricardo Erazo
# traveling waves in HH endogenous oscillatory neurons.

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


nr_neurons =1
ap1 = nr_neurons+1
tau2 = 3
Ie = 0.1
delta1 = 1e-2
sigma1 = 3e-2
gsyn= 20
domain = np.arange(0,nr_neurons)
# print domain
# Isyn = Ie2*np.exp(-(np.transpose(domain)*delta1/sigma1))
# temp1 = domain*delta1/sigma1
Ie2 = Ie*np.exp(-(domain*delta1/sigma1))
# print Isyn
Vt = 0.02 
Isyn = Ie2*np.exp(-np.arange(0,nr_neurons,1)*delta1/sigma1)
Ispike = Ie2*np.exp(-np.arange(0,nr_neurons,1)*delta1/sigma1)

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
pars = [tau2, Ie2, delta1, sigma1, gsyn, Htheta, K2theta, gNa, ENa, gK2, EK, gH, EH, gl, El, Vrev, Cm, tau_h, tau_k, tau_s, tau1, tau_m]
# ODE
#initial conditions
## rest
v0= -0.04224559
h0= 0.99240929
m0= 0.30086805
n0= 0.01758381
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
	Ie_time_dependent = Ie2*np.exp(-t/tau2)

	dv = (-1/Cm) * (gNa*mNass*mNass*mNass*y0[1]*(y0[0]-ENa)+ gK2*y0[3]*y0[3]*(y0[0]-EK) + gH*y0[2]*y0[2]*(y0[0]-EH) + gl*(y0[0]-El) + 0.006)
	dh = (1/(1+ np.exp(500*(y0[0]+0.0325))) - y0[1])/tau_h
	dm = (1/(1+2*np.exp(180*(y0[0]+Htheta))+np.exp(500*(y0[0]+Htheta))) -y0[2])/tau_m
	dn = (mK2ss - y0[3])/tau_k
	f = dv, dh, dm, dn
	return f


#Post synaptic neurons
def vecf_syn(t, y0):
	v,h,m,n = y0
	mNass=1./(1.+np.exp(-150.*(y0[0]+0.0305)))
	mK2ss=1./(1.+np.exp(-83.*(y0[0]+K2theta)))
	
	dv = (-1/Cm) * (gNa*np.power(mNass,3)*y0[1]*(y0[0]-ENa)+ gK2*np.power(y0[3],2)*(y0[0]-EK) + gH*np.power(y0[2],2)*(y0[0]-EH) + gl*(y0[0]-El) + 0.006 + gsyn*synapse(t)*(y0[0]-Vrev))
	dh = (1/(1+ np.exp(500*(y0[0]+0.0325))) - y0[1])/tau_h
	dm = (1/(1+2*np.exp(180*(y0[0]+Htheta))+np.exp(500*(y0[0]+Htheta))) -y0[2])/tau_m
	dn = (mK2ss - y0[3])/tau_k
	f = dv, dh, dm, dn
	return f

def Vt_cross(t,y0): return y0[0]-Vt

Vt_cross.terminal= True
Vt_cross.direction = 1

def initconds(x0,t):
	y0=[x0[0],x0[1],x0[2],x0[3]]
	yy0 = np.dot(y0,np.exp(-t/tau2))
	return yy0

def synapse(t):
	Isyn = Ie2*np.exp(-(t-ts)/tau2)
	return Isyn

xy0 = np.array([v0, h0, m0, n0])
xy2 = np.array([v2, h2, m2, n2])
t1= 0
tend = 10
#shocked neuron first.
shock = solve_ivp(vecf, [t1, tend], xy2, events= Vt_cross, method = 'RK45',rtol=1e-5,atol=1e-7)
ts = shock.t_events[0]

tspan = [ts, ts+10]
Ie_local = Isyn[0]
post1 = solve_ivp(vecf_syn, tspan, xy0, events = Vt_cross, method = 'RK45',rtol=1e-5,atol=1e-7)
Isyn = Isyn*np.exp(-(post1.t_events[0] - shock.t_events[0])/tau2)
ts = post1.t_events[0]
tspan = [ts, ts+10]
post2 = solve_ivp(vecf_syn, tspan, xy0, events = Vt_cross, method = 'RK45',rtol=1e-5,atol=1e-7)

plt.ion()
plt.plot(shock.t, shock.y[0,:], label='shock')
plt.plot(post1.t, post1.y[0,:], label='post')
plt.plot(post2.t, post2.y[0,:], label='post2')
plt.legend()
