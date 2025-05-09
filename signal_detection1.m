clear; clc; close all;

%% 1. Parameter Setup
N         = 128;                 % Length of the signal
Fs        = 1000;                % Sampling rate (Hz)
A         = 1;                   % Signal amplitude
numTrials = 3000;                % Number of Monte Carlo trials
SNR_dB    = -10:5:10;            % Range of SNR values 
rho       = 0.8;                 % AR(1) coefficient
Pfa_req   = 0.05;                % Desired false alarm probability
numThresh = 200;                 % Number of threshold points for ROC curves

%% 2. Template Signals
n = (0:N-1)';                        
s_pulse = hann(N);    s_pulse = s_pulse / norm(s_pulse);         % Hanning window pulse template
s_chirp = chirp(n/Fs,50,N/Fs,200); s_chirp = s_chirp / norm(s_chirp); % Linear chirp template
templates     = {s_pulse, s_chirp};
templateNames = {'Pulse','Chirp'};

%% 3. AR(1) Whitening Filter Coefficients
b = [1 -rho]; 
a = 1;    

%% 4. Preallocate Results Structure
numTmpl = numel(templates);
numSNR  = numel(SNR_dB);
results = repmat(struct( ...
    'Fraw',[],'Traw',[],'AUCRaw',[],'PDraw',[], ...
    'Fwh',[] ,'Twh',[] ,'AUCWh',[] ,'PDWh',[]), ...
    numTmpl, 2, numSNR);