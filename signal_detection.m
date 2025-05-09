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
n       = (0:N-1)';                        
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

%% 5. Monte Carlo Simulation
for ti = 1:numTmpl
    s_raw = templates{ti}(:);           % Raw template
    % Apply whitening filter to template and renormalize
    s_wh  = filter(b,a,s_raw);
    s_wh  = s_wh / norm(s_wh);
    
    for si = 1:numSNR
        % Compute noise variance for current SNR
        sigma2 = A^2 / 10^(SNR_dB(si)/10);
        % Initialize statistic arrays
        stat_raw0 = zeros(numTrials,1);
        stat_raw1 = zeros(numTrials,1);
        stat_wh0  = zeros(numTrials,1);
        stat_wh1  = zeros(numTrials,1);
        
        for k = 1:numTrials
            % Generate AR(1) colored noise
            w     = randn(N,1);
            n_col = filter(1,[1 -rho], sqrt(sigma2)*w);
            % Whiten the noise
            n_wh  = filter(b,a,n_col);
            
            % Construct received vectors under H0 and H1
            y0_raw = n_col;
            y1_raw = A*s_raw + n_col;
            y0_wh  = n_wh;
            y1_wh  = A*s_wh  + n_wh;
            
            % Compute matched filter outputs
            stat_raw0(k) = dot(s_raw, y0_raw);
            stat_raw1(k) = dot(s_raw, y1_raw);
            stat_wh0(k)  = dot(s_wh,  y0_wh);
            stat_wh1(k)  = dot(s_wh,  y1_wh);
        end
        
        % Compute ROC, AUC, and PD at specified Pfa for raw template
        [Fraw, Traw] = simpleROC(stat_raw0, stat_raw1, numThresh);
        AUCRaw       = abs(trapz(Fraw, Traw));
        thr_raw      = quantile(stat_raw0, 1-Pfa_req);
        PDraw        = mean(stat_raw1 > thr_raw);
        
        % Compute ROC, AUC, and PD at specified Pfa for whitened template
        [Fwh, Twh] = simpleROC(stat_wh0, stat_wh1, numThresh);
        AUCWh      = abs(trapz(Fwh, Twh));
        thr_wh     = quantile(stat_wh0, 1-Pfa_req);
        PDWh       = mean(stat_wh1 > thr_wh);
        
        % Store results in structure array
        results(ti,1,si) = struct('Fraw',Fraw,'Traw',Traw,'AUCRaw',AUCRaw,'PDraw',PDraw, ...
                                   'Fwh',[]  ,'Twh',[]   ,'AUCWh',[]  ,'PDWh',[]);
        results(ti,2,si) = struct('Fraw',[]     ,'Traw',[]     ,'AUCRaw',[]   ,'PDraw',[],   ...
                                   'Fwh',Fwh ,'Twh',Twh ,'AUCWh',AUCWh ,'PDWh',PDWh);
    end
end

%% 6. Plotting
% (1) ROC curves for Pulse template: Raw vs Whitened
figure('Name','ROC: Pulse Raw vs Whitened', ...
       'NumberTitle','off', 'Units','normalized', 'Position',[0.1,0.1,0.4,0.6]);
tiledlayout(3,2,'Padding','tight','TileSpacing','compact');
hRaw = []; hWh = [];
for si = 1:numSNR
    ax = nexttile; hold(ax,'on');
    h1 = plot(ax, results(1,1,si).Fraw, results(1,1,si).Traw);
    h2 = plot(ax, results(1,2,si).Fwh,  results(1,2,si).Twh);
    plot(ax, [0 1],[0 1],'k--'); % Reference diagonal line
    xlabel(ax,'False Alarm Rate');
    ylabel(ax,'Detection Probability');
    title(ax,sprintf('SNR = %d dB', SNR_dB(si)));
    axis(ax,[0 1 0 1]); grid(ax,'on');
    hold(ax,'off');
    if si == 1
        hRaw = h1; hWh = h2;  
    end
end
if numSNR < 6
    ax = nexttile(6); axis(ax,'off');
end
legend([hRaw hWh], {'Raw MF','Whitened MF'}, 'Orientation','horizontal', 'Location','southoutside');

% (2) ROC curves for Chirp template: Raw vs Whitened
figure('Name','ROC: Chirp Raw vs Whitened', ...
       'NumberTitle','off', 'Units','normalized', 'Position',[0.1,0.1,0.4,0.6]);
tiledlayout(3,2,'Padding','tight','TileSpacing','compact');
hRaw = []; hWh = [];
for si = 1:numSNR
    ax = nexttile; hold(ax,'on');
    h1 = plot(ax, results(2,1,si).Fraw, results(2,1,si).Traw);
    h2 = plot(ax, results(2,2,si).Fwh,  results(2,2,si).Twh);
    plot(ax, [0 1],[0 1],'k--');
    xlabel(ax,'False Alarm Rate'); ylabel(ax,'Detection Probability');
    title(ax,sprintf('SNR = %d dB', SNR_dB(si)));
    axis(ax,[0 1 0 1]); grid(ax,'on'); hold(ax,'off');
    if si == 1, hRaw = h1; hWh = h2; end
end
if numSNR < 6, ax = nexttile(6); axis(ax,'off'); end
legend([hRaw hWh], {'Raw MF','Whitened MF'}, 'Orientation','horizontal', 'Location','southoutside');

% (3) AUC vs SNR for both templates
figure('Name','AUC vs SNR','NumberTitle','off'); hold on;
markers = {'o','s'};
for ti = 1:numTmpl
    auc_raw = arrayfun(@(k) results(ti,1,k).AUCRaw, 1:numSNR);
    auc_wh  = arrayfun(@(k) results(ti,2,k).AUCWh,  1:numSNR);
    plot(SNR_dB, auc_raw, ['-' markers{ti}], 'DisplayName',[templateNames{ti} ' Raw']);
    plot(SNR_dB, auc_wh,  ['--' markers{ti}], 'DisplayName',[templateNames{ti} ' Whitened']);
end
xlabel('SNR (dB)'); ylabel('AUC'); legend('Location','best'); grid on;

%%function for ROC calculation
function [FPR, TPR] = simpleROC(s0, s1, numT)
    % Computes ROC curve given samples under H0 (s0) and H1 (s1)
    scores = [s0; s1];
    thr    = linspace(min(scores), max(scores), numT);
    FPR    = zeros(1, numT);
    TPR    = zeros(1, numT);
    for i = 1:numT
        FPR(i) = mean(s0 > thr(i));
        TPR(i) = mean(s1 > thr(i));
    end
end
