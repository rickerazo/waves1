%function HoHuxtwave
%simga influences the 
%erm
%Hodgkin Huxley Model, Traveling waves research
%Ricardo Erazo
%Fall 2016
clear all
close all
format long
% rand('state', 2)
global deltaN shockedNeurons
% addpath('/Users/RickEz/Dropbox/MATLAB/2016/Research_HHTW/initialConditions');
%% Network settings:
tFin = 50; % time limit (s) of system evolution
var= 5; %variables: V, hNa, mK2, mH, s
totalDomain = 1e0;%e-3;
deltaN = 5e-3; % size of neuron in meters -> 1 micron
% deltaN = 3e-5; % size of neuron in meters -> 3 microns

shockedLength = 0.2;
shockedNeurons = round(shockedLength/deltaN); % discrete number of neurons that will be excited as system evolves

restingNeurons = round((totalDomain - shockedLength) / deltaN);%discrete number: resting neurons. the total domain - shocked
nr_neurons = round(totalDomain / deltaN) %total neurons in the domain
neuronSpace = deltaN: deltaN : totalDomain; % continuous media: discrete deltaN vector

options_1 = odeset('Refine',1, 'RelTol', 1e-8, 'AbsTol', 1e-8);
options_2 = odeset('Refine', 1, 'RelTol', 1e-8,'AbsTol', 1e-8, 'Events', @events);
tspan= 0.: 5e-3: tFin; %vector - it tells matlab. this is in seconds, so use a smaller step maybe 0.1

% gsyn = 0.5;
% gsyn = 0.;
gsyn = 0.005;
sigma1 = 2; %[abstract] number of neurons connected to any given 'x' neuron
sigma2 = sigma1; %[practical] in terms of units of length, how far synaptic connections can reach out

J = zeros(nr_neurons);
for ii = 1: nr_neurons
    J(:, ii) = abs(neuronSpace - neuronSpace(ii));
end
J= deltaN*exp(-J/sigma1) / 2*sigma1;
J= J-diag(diag(J));
% one way connection ; for two way connection, comment out lines 33 to 35
for ii = 1:nr_neurons;
    J(ii, (ii :end) ) = 0;
end
%J is a matrix that holds the "strenght" of synaptic connection due to distance only
%deltaN *J
W = gsyn .* J;
%W is the product of J * gsyn. If gsyn = 1, J = W, but if gsyn > 1 connection is stronger than J(distance alone); if gsyn < 1 connection is weaker than J(distance alone)
r1 = J - W; % product difference. For evaluation purposes only
% figure; imagesc(J); colorbar; title('J')
% figure; imagesc(W); colorbar; title('W')
% figure; imagesc(r1); colorbar; title('J - W difference')

sprintf('%s', 'gsyn = ', num2str(max(max(W))), ' siemens.')
% 10e-3 mili
% 10e-6 micro
% 10e-9 nano
% 10e-12 pico
%% Parameters and initial conditions
%Parameters:
Cm=.5;
gNa=105.;
ENa=0.045;
gK2=30.;
EK=-0.07;      
gH=4.;
EH=-0.021;
gl=8.;
El=-0.046;
% Esyn=-0.0625; threshold for excitation between -0.2 and 0.1
% Esyn=0.001;
Esyn=0.01;

%Voltages of half activation
%10 s duration; 200 s interval
K2theta= -0.0075;
Htheta= 0.041326;
%6 s duration; 2 s interval
% K2theta= -0.0075;
% Htheta= 0.038;

%K2
%fig1 D=-0.0075; C= -0.0075 ; silent
%fig2 B=-0.0105, C=-0.0075, D=-0.0075, E=-0.0105
%fig4 A= -0.0069999 B=-0.0040999 C=0.005905 -- this; 
%H
%fig1 D= 0.038 ;C= 0.041326 ; silent
%fig 2 B=0.0413564925, C=0.041326, D=0.038, E=0.038
%fig4 A= 0.041319316864014 B= 0.041268055725098 C= 0.04073603515625 --  Htheta= 0.041326;

%initial conditions: variables
% load 'bursting_1C_1.mat' %initial conditions for bursting neurons, this is the population that is excited initially
% V1= zeros(1, nr_excited);
ii = 1;
%       firstNeuron: neuronSize : totalSchockedArea
for jj= deltaN: deltaN: shockedLength
% V1(1, ii) = V0(1);
% h1(1, ii) = h(1);
% mK1(1, ii) = mK(1);
% mH1(1, ii) = mH(1);
% s1(1, ii) = s(1);
%Initial conditions for bursting: upstroke of first action potential
V1(1, ii) = -0.026892644544022;
h1(1, ii) = 0.867431828696752;
mK1(1, ii) = 0.016203233894593;
mH1(1, ii) = 0.175309528389848;
s1(1, ii) = 6.255230227403558e-14;
ii = ii+1;
end
V0= V1; h= h1; mK= mK1; mH= mH1; s= s1;
    
% load 'resting.mat' %initial conditions for silent neurons, who form most of the network, and provide the network dynamics
% V2= zeros(1, nr_resting);
ii = 1;
%       first resting neuron: neuronSize : totalDomain
for jj= shockedLength + deltaN: deltaN: totalDomain
%     V03(1, ii) = V02(1);
%     h3(1, ii) = h2(1);
%     mK3(1, ii)= mK2(1);
%     mH3(1, ii)= mH2(1);
%     s3(1, ii)= s2(1);
    V03(1, ii) = -0.0420966;
    h3(1, ii) = 0.99179957;
    mK3(1, ii)= 0.016047153;
    mH3(1, ii)= 0.291963;
    s3(1, ii)= 3.88e-18;
ii = ii + 1;
end
V02= V03; h2= h3; mK2= mK3; mH2= mH3;  s2= s3;

V0= cat(1, V0', V02');
h= cat(1, h', h2');
mK= cat(1, mK', mK2');
mH= cat(1, mH', mH2');
s= cat(1, s', s2');

    yy0(:,1)=V0; %membrane potential
    yy0(:,2)=h; % Na inactivation
    yy0(:,3)=mK; %K activation
    yy0(:,4)=mH; %H activation
    yy0(:,5)=s; %s activation
    yy0;
%% ODE45
    %script, timespan, initial conditions, ODE options
    % [tspace, vspace, ty, ye, ie]=ode45(@HodgkinHuxley, tspan, yy0, options_2, ...
[tspace, vspace]=ode45(@HodgkinHuxley1, tspan, yy0, options_1, ...
    Cm, gNa, ENa, gK2, EK, gH, EH, gl, El, K2theta, Htheta, Esyn, W, var, nr_neurons);

%[datapoints, dataset]= size(vspace);

sprintf('%s', 'vspace is a matrix with all variables:')
sprintf('%s', 'Columns 1 to ', num2str(nr_neurons), ' = membrane potential')
sprintf('%s', 'Columns ', num2str(nr_neurons+1), ' to ', num2str(2*nr_neurons), ' = Na inactivation')
sprintf('%s', 'Columns ', num2str(2*nr_neurons+1), ' to ', num2str(3*nr_neurons), ' = K activation')
sprintf('%s', 'Columns ', num2str(3*nr_neurons+1), ' to ', num2str(4*nr_neurons), ' = hyperpolarization activated variable')
sprintf('%s', 'Columns ', num2str(4*nr_neurons+1), ' to ', num2str(5*nr_neurons), ' = synaptic activation')

% SpikeIdsBurstDiss
% BurstDissection
% save ('HHwave2', '-v7.3')
% sprintf('%s', 'max W = ', num2str(max(max(W))), ' siemens.')
% 10e-3 mili
% 10e-6 micro
% 10e-9 nano
% 10e-12 pico
% Vm = vspace(:, 1: nr_neurons);
% end
% 
% % function [value,isterminal,direction] = events(tspan, vspace)
% % th = vspace(:, 1);
% % direction= 1;
% % value= th;
% % isterminal= +1;
% % end
% function [value,isterminal,direction] = events(tspan, vspace);
% th1= vspace(1, 1);
% th2= vspace(2, 1);
% direction= [1, 1];
% value= [th1, th2];
% isterminal=[0, 0, 0];
% end
