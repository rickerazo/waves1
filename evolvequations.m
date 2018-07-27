function dy =HodgkinHuxley(t,y,...
Cm, gNa, ENa, gK2, EK, gH, EH, gl, El, K2theta, Htheta, Esyn, W, var, nr_neurons)

u= reshape(y, nr_neurons, var);
s_slope = -1500;%1000
%Steady states:
%mNa steadystate (V) - instantaneous INa activation
mNass=1./(1.+exp(-150.*(u(:,1)+0.0305)));
%mK2 steadystate (V,K2theta)
mK2ss=1./(1.+exp(-83.*(u(:,1)+K2theta)));
%s steady state --- sss=1./(1.+exp(-sB.*(u(1,:)+sV12)));
sss=1./(1.+exp(s_slope.*(u(:,1)+0.002)));

%V
dy(1:nr_neurons,1)=-(1./Cm).*(... %Membrane potential
+gNa.*mNass.*mNass.*mNass.*u(:,2).*(u(:,1)-ENa)... %gNa*mNa(V)^3*hNa*(V-ENa)
+gK2.*u(:,3).*u(:,3).*(u(:,1)-EK)...%gK2*mK2^2*(V-EK)
+gH.*u(:,4).*u(:,4).*(u(:,1)-EH)...%gH*mH^2*(V-EH)
+gl.*(u(:,1)-El)+0.006...
+(W*u(:,5)));%gsyn*synaptic function
% +(W*u(:,5)).*(u(:,1)-Esyn));

% dy(1:nr_neurons,1)=du;%1
%1:5
%hNa
dy(nr_neurons+1:2*nr_neurons,1)=(1./(1.+exp(500.*(u(:,1)+0.0325)))-u(:,2))./0.0405;
% dy(nr_neurons+1:2*nr_neurons,1)=du;
%6:10
%mK2
dy(2*nr_neurons+1:3*nr_neurons,1)=(mK2ss-u(:,3))./2;
% dy(2*nr_neurons+1:3*nr_neurons,1)=du;
%11:15
%mH
dy(3*nr_neurons+1:4*nr_neurons,1)=(1./(1.+2.*exp(180.*(u(:,1)+Htheta))+exp(500.*(u(:,1)+Htheta)))-u(:,4))./0.1;
% dy(3*nr_neurons+1:4*nr_neurons,1)=du;
%16:20
%s
dy(4*nr_neurons+1:5*nr_neurons,1)=(sss-u(:,5))./0.1;
% dy(4*nr_neurons+1:5*nr_neurons,1)=du;clo s
%synapse time constant: rise and decay constants
end
